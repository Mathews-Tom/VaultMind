# ADR-0002: Integration Capability and Outbound Approval Model

**Status:** Accepted
**Date:** 2026-04-02
**Author:** Codex

## Context

VaultMind is positioned to add external integrations, proactive orchestration, and outbound assistance. Those features introduce a high-risk surface where private data access, untrusted content, and outbound effects can overlap. If provider adapters are added without a stable capability model, each integration will invent its own rules for reading, drafting, writing, and sending, which will make the system difficult to audit and easy to misuse.

The system also needs a durable artifact model for outbound help. Ephemeral suggestions in chat are not sufficient when drafts need review, approval, expiry handling, retrieval for tone matching, and auditability.

## Decision

VaultMind will standardize external integration permissions around capability tiers and separate draft generation from external send operations. The initial shared capability tiers are `read`, `draft`, and `write`, with explicit confirmation required for external send and destructive actions. Future provider adapters must inherit these boundaries rather than defining their own trust model.

## Alternatives Considered

### Give each integration its own permission semantics
- **Pros:** Maximum flexibility for each provider.
- **Cons:** Inconsistent behavior, duplicated policy logic, and weak auditability.
- **Rejected because:** Shared policy is more important than provider-specific convenience.

### Start with full write access for early integrations
- **Pros:** Faster demos and broader automation from day one.
- **Cons:** Higher risk, weaker operator trust, and more difficult rollback if prompts or adapters misbehave.
- **Rejected because:** VaultMind should start from the narrowest safe contract and widen later.

### Skip draft artifacts and rely on transient chat output
- **Pros:** Less storage and fewer lifecycle rules.
- **Cons:** No durable review path, no voice-matching corpus, no expiry handling, and poor audit support.
- **Rejected because:** Drafts are a first-class product surface, not just a prompt output.

## Consequences

### Positive
- Integration work can scale without re-litigating the trust model for every provider.
- Draft creation, approval, and send paths stay inspectable and auditable.
- External actions can be added behind explicit human approval gates.

### Negative
- The first integrations will look conservative and slower than fully autonomous demos.
- Provider adapters must conform to shared contracts, which adds implementation discipline.

### Neutral
- The capability tiers are architectural defaults and can be extended later if a concrete need emerges.
- Existing VaultMind features remain unaffected until integration runtime code is added.

## References

- `.docs/vaultmind-enhancement-plan.md`
- `docs/architecture.md`
- `src/vaultmind/config.py`
