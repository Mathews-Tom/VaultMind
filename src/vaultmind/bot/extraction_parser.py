"""Parser for inline XML extraction tags from LLM thinking responses."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

NODE_TYPES: frozenset[str] = frozenset(
    {"person", "project", "concept", "tool", "organization", "event", "location"}
)
EDGE_TYPES: frozenset[str] = frozenset(
    {
        "related_to",
        "part_of",
        "depends_on",
        "created_by",
        "influences",
        "mentioned_in",
        "competes_with",
        "preceded_by",
    }
)
EPISODE_STATUSES: frozenset[str] = frozenset(
    {"pending", "success", "failure", "partial", "unknown"}
)


@dataclass
class ExtractedEntity:
    name: str
    type: str
    confidence: float = 1.0
    description: str = ""


@dataclass
class ExtractedRelationship:
    source: str
    target: str
    relation_type: str
    confidence: float = 1.0


@dataclass
class ExtractedEpisode:
    decision: str
    context: str = ""
    outcome: str = ""
    status: str = "pending"
    confidence: float = 1.0
    lessons: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    clean_response: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    episodes: list[ExtractedEpisode] = field(default_factory=list)


_RE_ENTITY = re.compile(r"<vm:entity\b[^>]*>.*?</vm:entity>", re.DOTALL)
_RE_RELATIONSHIP_PAIRED = re.compile(r"<vm:relationship\b[^>]*>.*?</vm:relationship>", re.DOTALL)
_RE_RELATIONSHIP_SELF = re.compile(r"<vm:relationship\b[^>]*/\s*>")
_RE_EPISODE = re.compile(r"<vm:episode\b[^>]*>.*?</vm:episode>", re.DOTALL)

_RE_STRIP_SELF_CLOSING = re.compile(r"<vm:\w+[^>]*/\s*>")
_RE_STRIP_PAIRED = re.compile(r"<vm:\w+[^>]*>.*?</vm:\w+>", re.DOTALL)
_RE_STRIP_CLOSING = re.compile(r"</vm:\w+>")
_RE_COLLAPSE_NEWLINES = re.compile(r"\n{3,}")
_RE_VM_PREFIX = re.compile(r"<(/?)vm:")


def _strip_namespace(xml_str: str) -> str:
    """Strip vm: namespace prefix so ET.fromstring can parse the tag."""
    return _RE_VM_PREFIX.sub(r"<\1", xml_str)


def _parse_confidence(raw: str | None) -> float:
    """Parse confidence string, clamp to [0.0, 1.0], default 1.0."""
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return max(0.0, min(1.0, value))


def _parse_entity(xml_str: str) -> ExtractedEntity | None:
    try:
        elem = ET.fromstring(_strip_namespace(xml_str))
    except ET.ParseError:
        logger.warning("Malformed vm:entity tag: %s", xml_str[:120])
        return None

    name = elem.get("name")
    if not name:
        logger.warning("vm:entity missing 'name' attribute")
        return None

    entity_type = elem.get("type")
    if entity_type not in NODE_TYPES:
        logger.warning("vm:entity invalid type '%s'", entity_type)
        return None

    confidence = _parse_confidence(elem.get("confidence"))
    description = (elem.text or "").strip()

    return ExtractedEntity(
        name=name,
        type=entity_type,
        confidence=confidence,
        description=description,
    )


def _parse_relationship(xml_str: str) -> ExtractedRelationship | None:
    try:
        elem = ET.fromstring(_strip_namespace(xml_str))
    except ET.ParseError:
        logger.warning("Malformed vm:relationship tag: %s", xml_str[:120])
        return None

    source = elem.get("from")
    target = elem.get("to")
    if not source or not target:
        logger.warning("vm:relationship missing 'from' or 'to' attribute")
        return None

    if source == target:
        logger.warning("vm:relationship self-edge skipped: '%s'", source)
        return None

    rel_type = elem.get("type")
    if rel_type not in EDGE_TYPES:
        logger.warning("vm:relationship invalid type '%s'", rel_type)
        return None

    confidence = _parse_confidence(elem.get("confidence"))

    return ExtractedRelationship(
        source=source,
        target=target,
        relation_type=rel_type,
        confidence=confidence,
    )


def _parse_episode(xml_str: str) -> ExtractedEpisode | None:
    try:
        elem = ET.fromstring(_strip_namespace(xml_str))
    except ET.ParseError:
        logger.warning("Malformed vm:episode tag: %s", xml_str[:120])
        return None

    decision = elem.get("decision")
    if not decision:
        logger.warning("vm:episode missing 'decision' attribute")
        return None

    context = elem.get("context", "")
    outcome = elem.get("outcome", "")
    status = elem.get("status", "pending")
    if status not in EPISODE_STATUSES:
        status = "pending"

    confidence = _parse_confidence(elem.get("confidence"))

    lessons: list[str] = []
    for lesson_elem in elem.findall("lesson"):
        text = (lesson_elem.text or "").strip()
        if text:
            lessons.append(text)

    entities: list[str] = []
    for entity_elem in elem.findall("entity"):
        text = (entity_elem.text or "").strip()
        if text:
            entities.append(text)

    return ExtractedEpisode(
        decision=decision,
        context=context,
        outcome=outcome,
        status=status,
        confidence=confidence,
        lessons=lessons,
        entities=entities,
    )


def _strip_tags(raw_response: str) -> str:
    """Remove all vm: tags and collapse excessive newlines."""
    clean = _RE_STRIP_SELF_CLOSING.sub("", raw_response)
    clean = _RE_STRIP_PAIRED.sub("", clean)
    clean = _RE_STRIP_CLOSING.sub("", clean)
    clean = _RE_COLLAPSE_NEWLINES.sub("\n\n", clean)
    return clean.strip()


def parse_extraction_tags(raw_response: str) -> ExtractionResult:
    """Parse XML extraction tags from LLM response.

    Returns ExtractionResult with clean_response (tags stripped) and
    parsed entities, relationships, and episodes. Malformed tags are
    logged and skipped; never raises on bad input.
    """
    entities: list[ExtractedEntity] = []
    for match in _RE_ENTITY.finditer(raw_response):
        parsed = _parse_entity(match.group(0))
        if parsed is not None:
            entities.append(parsed)

    relationships: list[ExtractedRelationship] = []
    for match in _RE_RELATIONSHIP_SELF.finditer(raw_response):
        parsed_rel = _parse_relationship(match.group(0))
        if parsed_rel is not None:
            relationships.append(parsed_rel)
    for match in _RE_RELATIONSHIP_PAIRED.finditer(raw_response):
        parsed_rel = _parse_relationship(match.group(0))
        if parsed_rel is not None:
            relationships.append(parsed_rel)

    episodes: list[ExtractedEpisode] = []
    for match in _RE_EPISODE.finditer(raw_response):
        parsed_ep = _parse_episode(match.group(0))
        if parsed_ep is not None:
            episodes.append(parsed_ep)

    clean_response = _strip_tags(raw_response)

    return ExtractionResult(
        clean_response=clean_response,
        entities=entities,
        relationships=relationships,
        episodes=episodes,
    )
