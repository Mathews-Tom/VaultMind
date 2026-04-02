# ADR-0001: Curated Memory and Autonomy Boundaries

**Status:** Accepted
**Date:** 2026-04-02
**Author:** Codex

## Context

VaultMind already has strong retrieval, graph, episodic memory, and scheduled loop foundations. What it does not yet have is a stable architectural boundary for high-signal memory that should always be available, or a single policy model that defines how proactive the system is allowed to be. Without those boundaries, later implementation risks spreading memory rules across chat handlers, loop jobs, and future integrations.

The system also needs a clear separation between raw interaction history, curated durable memory, and decision-outcome records. Treating all memory as one bucket would create prompt bloat, unclear ownership, and duplication between the vault, SQLite stores, and graph-derived context.

## Decision

VaultMind will adopt a four-part memory model and a global autonomy policy. The memory model consists of identity memory, retrieval memory, episodic memory, and promoted memory. The autonomy policy consists of four modes: `observer`, `advisor`, `assistant`, and `partner`, defined centrally in configuration and consumed by later runtime components.

## Alternatives Considered

### Keep the current implicit model
- **Pros:** No schema expansion, no new documentation, no migration work.
- **Cons:** Future behavior would be scattered across handlers and jobs, and memory responsibilities would stay ambiguous.
- **Rejected because:** The next implementation steps need stable boundaries before behavior is added.

### Collapse all memory into retrieval and session summaries
- **Pros:** Minimal architecture, fewer explicit storage concepts.
- **Cons:** Durable preferences and priorities would depend on retrieval ranking, and prompt assembly would remain noisy.
- **Rejected because:** Always-available context needs stronger guarantees than retrieval can provide.

### Make autonomy decisions locally inside each subsystem
- **Pros:** Individual subsystems can evolve independently.
- **Cons:** Policy drift is likely, and users would face inconsistent trust boundaries across features.
- **Rejected because:** VaultMind needs one source of truth for proactive behavior.

## Consequences

### Positive
- Future memory work can land against explicit storage roles instead of inferred conventions.
- Proactive behavior can be gated consistently across bot flows, scheduler jobs, and integrations.
- Documentation and config now describe the target system shape before runtime changes start.

### Negative
- The configuration surface becomes larger before all runtime features exist.
- Some terms such as identity memory and promoted memory add conceptual overhead that the implementation must justify.

### Neutral
- Existing behavior is unchanged by this decision alone.
- Additional runtime modules will still be needed to realize the full design.

## References

- `.docs/vaultmind-enhancement-plan.md`
- `docs/architecture.md`
- `src/vaultmind/config.py`
