# Axiom Runtime: Scope and Non-Goals

## 1. Vision & Problem Statement

Production agent workflows frequently fail due to unhandled transient faults, unbounded retry loops causing duplicate side-effects (e.g. executing payments or external API calls twice), lost execution context across process restarts, and absent human-in-the-loop policy controls.

**Axiom Runtime** is an open-source, deterministic, stateful agent execution framework engineered to turn brittle agent prototypes into resilient, observable production software.

---

## 2. In-Scope Capabilities

1. **Durable State Machine Execution**:
   - Explicit, deterministic state transitions (`CREATED`, `RUNNING`, `WAITING_FOR_TOOL`, `WAITING_FOR_APPROVAL`, `EXECUTING_TOOL`, `CHECKPOINTED`, `RETRYING`, `COMPLETED`, `FAILED`, `CANCELLED`).
   - Persisted checkpoints before and after every external tool execution and model generation step.

2. **Idempotency & Replay Protection**:
   - Strict tool execution deduplication using SHA-256 idempotency keys calculated as `sha256(run_id:step_index:tool_name:canonical_json(arguments))`.
   - Replay of verified results without re-executing external side-effects when recovering from worker crashes or process interruptions.

3. **Policy-Gated Tool Execution & HITL**:
   - Action policy engine providing `ALLOW`, `REQUIRE_APPROVAL`, and `DENY` gating.
   - First-class human-in-the-loop (HITL) approval with timeout deadlines and audit logs.

4. **Fault Classification & Recovery**:
   - Structured error taxonomies (`MODEL_TIMEOUT`, `TOOL_SCHEMA_ERROR`, `EXTERNAL_TOOL_5XX`, `POLICY_BLOCKED`, `APPROVAL_EXPIRED`, `STATE_CONFLICT`).
   - Bounded exponential backoff retries with optimistic concurrency control (`version` checking).

5. **Tool Registry & MCP Integration**:
   - Pydantic v2 schema-validated tools.
   - Model Context Protocol (MCP) compatible tool declarations.

---

## 3. Explicit Non-Goals

1. **General Chatbot / Consumer UI Framework**:
   - Axiom Runtime is not a chatbot frontend, prompt playground, or UI component library. It provides the backend execution engine and API layer.

2. **Proprietary Vendor Lock-In**:
   - Axiom Runtime is model-agnostic and does not hardcode reliance on a single proprietary LLM provider; all adapters conform to a unified `ModelAdapter` interface.

3. **Complex Multi-Agent Swarm Emergence**:
   - Axiom avoids unconstrained, autonomous multi-agent chatter. Instead, multi-agent interactions are modeled as explicit, deterministic workflows with auditable state transitions.

4. **In-Memory-Only Ephemeral Scratchpads**:
   - Axiom requires durable persistence. Ephemeral state that vanishes on process termination is intentionally unsupported for production run execution.
