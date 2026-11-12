# Axiom Runtime: Scope and Non-Goals

## 1. Problem Statement
Most AI agent prototypes succeed in interactive demos yet fail in production environments due to seven systemic reliability gaps:
1. **Lost Execution State**: Agent progress is kept in volatile memory and lost across process restarts or sessions.
2. **Unrecovered Tool Failures**: Transient network errors or API timeouts abort the entire run without structured recovery.
3. **Non-Resumable Long-Running Tasks**: Multi-step workflows that pause for external events or human review cannot cleanly serialize and resume.
4. **Unguarded Side Effects**: Agents invoke mutating or high-risk tools without policy checks or human-in-the-loop (HITL) approval gates.
5. **Concurrency Races**: Duplicate webhook deliveries or parallel worker attempts corrupt shared task state.
6. **Opaque Failures**: Developers lack fine-grained step traces, token/latency metrics, and deterministic failure classifications.
7. **Duplicate Side Effects on Retry**: Retrying a crashed step after an external tool has already succeeded repeats the real-world side effect.

## 2. Product Proposition
**Axiom Runtime** is a lightweight, modular execution engine for stateful AI agents providing:
- **Durable State Transitions**: Every step transition and checkpoint is persisted atomically.
- **Idempotent Tool Execution**: Deterministic idempotency keys (`sha256(run_id, step_index, tool_name, canonical_args)`) ensure retried steps replay completed tool outputs instead of repeating external mutations.
- **Policy & Human Approval Gates**: Pre-execution policy hooks enforce `ALLOW`, `REQUIRE_APPROVAL`, and `DENY` decisions with configurable deadlines.
- **Structured Observability**: Step-level execution traces, token/cost accounting, and failure classification out of the box.

## 3. In-Scope (v0.1 – v1.0)
- Deterministic execution state machine (`CREATED`, `RUNNING`, `WAITING_FOR_TOOL`, `WAITING_FOR_APPROVAL`, `EXECUTING_TOOL`, `CHECKPOINTED`, `RETRYING`, `COMPLETED`, `FAILED`, `CANCELLED`).
- Pluggable persistence layer supporting both durable SQL (`PostgreSQL` / `SQLite`) and deterministic in-memory storage for fast testing.
- Provider-agnostic `ModelAdapter` gateway (supporting Gemini, OpenAI, Anthropic, and deterministic fault-injection mocks).
- Typed `ToolRegistry` with Pydantic v2 schema validation, idempotency deduplication, and Model Context Protocol (MCP) adapter compatibility.
- Bounded exponential backoff retry policies distinguishing transient errors from terminal policy or validation errors.
- First-class Fault-Injection Verification Suite covering all six production failure modes.

## 4. Explicit Non-Goals
- **Not a Prompt Engineering or UI Library**: Axiom Runtime focuses strictly on the backend execution lifecycle, state durability, and control plane—not chat UI widgets or prompt templates.
- **Not a Replacement for Distributed Workflow Engines at Hyperscale**: The goal is not to out-scale Temporal or Kubernetes, but to provide a clear, inspectable, agent-native execution kernel with first-class LLM/tool semantics.
- **No Premature Microservices**: Axiom Runtime starts as a **modular monolith** with clean subsystem interfaces (`Engine`, `State`, `Gateway`, `Tools`, `Policy`, `Scheduler`, `Telemetry`). Distributed Redis workers are optional adapters introduced only when horizontal scaling is required.
