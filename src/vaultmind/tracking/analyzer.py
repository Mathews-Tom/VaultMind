"""Preference pattern analysis — generates insights from tracked interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from vaultmind.tracking.preferences import InteractionType, PreferenceStore


@dataclass(frozen=True, slots=True)
class PreferenceInsights:
    total_interactions: int
    period_days: int
    top_searches: list[tuple[str, int]]
    top_tags_approved: list[tuple[str, int]]
    top_tags_rejected: list[tuple[str, int]]
    capture_topics: list[tuple[str, int]]
    interaction_counts: dict[str, int]
    suggestions_acceptance_rate: float
    active_hours: list[int]
    recommendations: list[str] = field(default_factory=list)


def analyze_preferences(
    store: PreferenceStore,
    days: int = 30,
) -> PreferenceInsights:
    """Analyze user interaction patterns and generate insights."""
    since = datetime.now() - timedelta(days=days)

    counts = store.get_counts(since=since)
    total = sum(counts.values())
    top_searches = store.get_top_searches(limit=20, since=since)
    top_approved = store.get_top_tags(approved=True, limit=20)
    top_rejected = store.get_top_tags(approved=False, limit=20)
    capture_topics = store.get_capture_topics(limit=20, since=since)
    active_hours = store.get_active_hours(since=since)

    # Suggestion acceptance rate
    accepted = counts.get(InteractionType.SUGGESTION_ACCEPTED, 0)
    rejected = counts.get(InteractionType.SUGGESTION_REJECTED, 0)
    suggestion_total = accepted + rejected
    acceptance_rate = accepted / suggestion_total if suggestion_total > 0 else 0.0

    # Generate recommendations
    recommendations = _generate_recommendations(
        top_searches=top_searches,
        top_rejected=top_rejected,
        active_hours=active_hours,
        acceptance_rate=acceptance_rate,
        suggestion_total=suggestion_total,
        counts=counts,
    )

    return PreferenceInsights(
        total_interactions=total,
        period_days=days,
        top_searches=top_searches,
        top_tags_approved=top_approved,
        top_tags_rejected=top_rejected,
        capture_topics=capture_topics,
        interaction_counts={k.value: v for k, v in counts.items()},
        suggestions_acceptance_rate=acceptance_rate,
        active_hours=active_hours,
        recommendations=recommendations,
    )


def _generate_recommendations(
    *,
    top_searches: list[tuple[str, int]],
    top_rejected: list[tuple[str, int]],
    active_hours: list[int],
    acceptance_rate: float,
    suggestion_total: int,
    counts: dict[InteractionType, int],
) -> list[str]:
    """Generate human-readable preference insights."""
    recs: list[str] = []

    # Frequent searches → suggest MOC notes
    for query, count in top_searches[:3]:
        if count >= 3:
            recs.append(
                f"You frequently search for '{query}' ({count}x) "
                f"-- consider creating a Map of Content (MOC) note for this topic."
            )

    # Rejected tags → suggest exclusions
    for tag, count in top_rejected[:3]:
        if count >= 2:
            recs.append(
                f"You reject the tag '{tag}' often ({count}x) "
                f"-- consider adding it to auto-tag exclusions."
            )

    # Active hours → digest timing
    if active_hours:
        top_hour = active_hours[0]
        recs.append(
            f"Your most active hour is {top_hour}:00 -- digest delivery could be scheduled then."
        )

    # Suggestion acceptance rate
    if suggestion_total >= 5:
        pct = int(acceptance_rate * 100)
        if acceptance_rate >= 0.8:
            recs.append(
                f"You accept {pct}% of link suggestions "
                f"-- the suggestion threshold is well-calibrated."
            )
        elif acceptance_rate < 0.4:
            recs.append(
                f"You accept only {pct}% of link suggestions "
                f"-- consider raising the similarity threshold."
            )

    # URL ingestion patterns
    url_count = counts.get(InteractionType.URL_INGESTED, 0)
    if url_count >= 10:
        recs.append(
            f"You've ingested {url_count} URLs -- consider running "
            f"'vaultmind research' to analyze them as a batch."
        )

    return recs


def generate_preference_report(insights: PreferenceInsights) -> str:
    """Generate a formatted markdown report from insights."""
    lines: list[str] = [
        "# VaultMind Usage Insights",
        "",
        f"**Period:** Last {insights.period_days} days"
        f" | **Total interactions:** {insights.total_interactions}",
        "",
    ]

    # Search patterns
    if insights.top_searches:
        lines.append("## Search Patterns")
        lines.append("")
        lines.append("| Query | Count |")
        lines.append("|-------|-------|")
        for query, count in insights.top_searches:
            lines.append(f"| {query} | {count} |")
        lines.append("")

    # Tag preferences
    if insights.top_tags_approved or insights.top_tags_rejected:
        lines.append("## Tag Preferences")
        lines.append("")
        if insights.top_tags_approved:
            lines.append("### Approved")
            lines.append("")
            lines.append("| Tag | Count |")
            lines.append("|-----|-------|")
            for tag, count in insights.top_tags_approved:
                lines.append(f"| {tag} | {count} |")
            lines.append("")
        if insights.top_tags_rejected:
            lines.append("### Rejected")
            lines.append("")
            lines.append("| Tag | Count |")
            lines.append("|-----|-------|")
            for tag, count in insights.top_tags_rejected:
                lines.append(f"| {tag} | {count} |")
            lines.append("")

    # Capture topics
    if insights.capture_topics:
        lines.append("## Capture Topics")
        lines.append("")
        lines.append("| Topic | Count |")
        lines.append("|-------|-------|")
        for topic, count in insights.capture_topics:
            lines.append(f"| {topic} | {count} |")
        lines.append("")

    # Activity distribution
    if insights.interaction_counts:
        lines.append("## Activity Distribution")
        lines.append("")
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for itype, count in sorted(insights.interaction_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {itype} | {count} |")
        lines.append("")

    # Active hours
    if insights.active_hours:
        top_hours = insights.active_hours[:5]
        hours_str = ", ".join(f"{h}:00" for h in top_hours)
        lines.append("## Active Hours")
        lines.append("")
        lines.append(f"Top active hours: {hours_str}")
        lines.append("")

    # Suggestions
    if insights.suggestions_acceptance_rate > 0:
        pct = int(insights.suggestions_acceptance_rate * 100)
        lines.append(f"## Suggestion Acceptance Rate: {pct}%")
        lines.append("")

    # Recommendations
    if insights.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for rec in insights.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)
