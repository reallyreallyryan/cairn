# cairn — AI Agent Guide

Instructions for AI models working on this codebase.

## Project Overview

cairn is a LangGraph-based AI agent with 16 tools, a Supabase/pgvector memory store (SCMS), Docker sandbox for code execution, a metatool system for dynamic tool creation, a task queue with daemon mode, an MCP server for cloud access, and a daily research digest pipeline. The agent runs locally via CLI; the MCP server is deployed on Railway.

## Project Structure

```
agent/                  # Core agent logic
  graph.py              # LangGraph StateGraph: START → classify → plan → act → reflect → END
  classifier.py         # Deterministic keyword-based task classification (no LLM call)
  classify.py           # CLASSIFY node: task type detection, project detection, SCMS context
  plan.py               # PLAN node: LLM plan generation, step parsing (parse_plan_steps)
  act.py                # ACT node: tool execution, FALLBACK_DISPATCH, keyword fallback
  reflect.py            # REFLECT node: result evaluation, continuation, SCMS decision logging
  utils.py              # Shared utilities: get_llm(), clean_output()
  nodes.py              # Re-exports from split modules (backward compat for graph.py)
  state.py              # AgentState TypedDict + PlanStep TypedDict
  model_router.py       # Complexity classification → tier selection → budget check
  daemon.py             # Background task queue processor with signal handling
  digest.py             # Daily research digest pipeline orchestrator
  evaluation.py         # Digest evaluation pipeline: approval/rejection metrics + threshold analysis
  notifications.py      # macOS osascript + file log notifications
  tools/
    __init__.py          # TOOL_REGISTRY dict, CATEGORY_TOOLS mapping, load_approved_custom_tools()
    web_search.py        # DuckDuckGo search
    url_reader.py        # Webpage content extraction (trafilatura → BeautifulSoup fallback)
    arxiv_search.py      # arXiv API search
    github_search.py     # GitHub API search
    file_reader.py       # Local file reading (restricted to allowed_directories)
    file_writer.py       # Local file writing (restricted to allowed_directories)
    note_taker.py        # Markdown note creation + SCMS storage
    code_executor.py     # Docker sandbox execution (subprocess fallback requires --allow-subprocess)
    scms_tools.py        # scms_search and scms_store wrappers
    project_tools.py     # create_project, update_project, archive_project
    metatool.py          # create_tool, test_tool, list_pending_tools
    custom/              # Directory for metatool-generated tool files

mcp_server/
  __init__.py
  config.py              # MCPSettings: mcp_base_url, mcp_host, mcp_port
  server.py              # FastMCP server: 16 MCP tools, OAuth 2.1 via InMemoryOAuthProvider

config/
  settings.py            # Pydantic BaseSettings, loads from .env
  model_routing.yaml     # Tier definitions (4 tiers) + routing rules + budget config
  digest_sources.yaml    # Digest pipeline source URLs, frequencies, relevance settings
  sandbox_policy.yaml    # Docker resource limits + security policy

sandbox/
  Dockerfile             # python:3.11-slim ARM64, non-root user, pre-installed packages
  manager.py             # SandboxManager: container lifecycle, code injection, cleanup

scms/
  client.py              # SCMSClient: memories, projects, decisions, tools, task queue CRUD
  embeddings.py          # Embedding generation via Ollama or OpenAI
  migrations/
    001_initial.sql      # projects, memories (pgvector), tool_registry, decision_log tables
    002_sandbox_metatool.sql  # approval_status, source_code columns on tool_registry
    003_task_queue.sql   # task_queue table with priority, recurring, cost tracking
    004a_embedding_pre_migrate.sql   # Drop old 768-dim HNSW index before dimension change
    004b_embedding_post_migrate.sql  # Alter embedding to vector(1536), rebuild HNSW index, update RPC
    005_add_archived_status.sql      # Add 'archived' to projects_status_check constraint

tests/
  test_project_crud.py    # SCMSClient CRUD, MCP tool wrappers, classifier routing
  test_metatool_loading.py  # Integration test: custom tools get @tool decorator on load
  test_digest_dedup.py    # Digest queue dedup, extract helpers, cancelled completed_at fix
  test_digest_fewshot.py  # Few-shot calibration from approval/rejection history
  test_evaluation.py      # Digest eval pipeline: parsing, metrics, bucketing, report

main.py                  # CLI entry point (argparse), initial_state construction, graph invocation
Dockerfile.mcp           # python:3.12-slim, uv-managed deps, runs mcp_server.server
Procfile                 # Railway entrypoint: uv run python -m mcp_server.server
railway.toml             # Railway build config: Dockerfile.mcp, health check
```

## Architecture: Classify → Plan → Act → Reflect

The agent runs as a LangGraph StateGraph with 4 nodes in a loop:

1. **Classify** (`classify_node`): Keyword matching determines `task_type` (research, knowledge_management, productivity, technical, multi, metatool). Detects project name via exact/fuzzy match. Retrieves context from SCMS (project info, recent memories, relevant decisions). No LLM call.

2. **Plan** (`plan_node`): LLM generates 1-3 numbered steps. Each step is parsed into a `PlanStep` with an inferred `tool_hint`. Model selected via `route_and_get_llm()`.

3. **Act** (`act_node`): Binds available tools to LLM via `bind_tools()`. If LLM produces structured tool calls, executes them. If not, falls back to keyword-based dispatch (`FALLBACK_DISPATCH` dict in nodes.py). Auto-saves search results to SCMS when save-related keywords are present.

4. **Reflect** (`reflect_node`): Marks step completed. Checks for redundancy (same tool 3+ times, repeated results). LLM synthesizes answer. Decides CONTINUE or COMPLETE. Logs decision to SCMS. Loop continues if more steps remain and iteration < max_iterations (10).

## State Schema (agent/state.py)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    task_type: str          # "research"|"knowledge_management"|"productivity"|"technical"|"multi"
    project: str
    available_tools: list[str]
    context: str            # SCMS-retrieved context
    plan: str
    plan_steps: list[PlanStep]  # [{step, tool_hint, status}]
    current_step: int
    step_results: dict[int, str]
    tools_used: list[str]
    decisions: list[dict]
    pending_tools: list[dict]
    sandbox_logs: list[dict]
    result: str
    should_continue: bool
    model_used: str
    cost_estimate: float
    iteration: int
```

## TOOL_REGISTRY Pattern

All tools are registered in `agent/tools/__init__.py`:

```python
TOOL_REGISTRY = {
    "tool_name": {
        "tool": <langchain @tool function>,
        "categories": ["research", "technical", ...],
        "keywords": ["search", "find", ...]
    },
    ...
}
```

`CATEGORY_TOOLS` maps each category to its list of tool callables. The classifier uses `CATEGORY_TOOLS` to select which tools are available for a given task type.

At startup, `load_approved_custom_tools()` dynamically imports approved metatool-generated tools from `agent/tools/custom/` using importlib and adds them to the registry. Bare functions (without `@tool` decorator) are auto-wrapped with `@tool` on load to ensure LangGraph compatibility.

## How to Add a New Built-in Tool

1. Create `agent/tools/your_tool.py` with a `@tool` decorated function
2. Import it in `agent/tools/__init__.py`
3. Add an entry to `TOOL_REGISTRY` with categories and keywords
4. Add the tool callable to relevant lists in `CATEGORY_TOOLS`
5. Add keywords to `agent/classifier.py` if creating a new category
6. If tool needs new state fields, add them to `AgentState` in `state.py` and `initial_state` in `main.py`

## Model Routing

`agent/model_router.py` reads `config/model_routing.yaml`:
- **Tiers**: local_light (Qwen 3 8B, $0), local (Qwen 3 32B, $0), cloud_standard ($0.01/call), cloud_advanced ($0.03/call)
- **Rules**: Checked in order. Each rule matches on task_type + keywords → routes to a tier. `simple_recall`/`simple_notes` → local_light, `summarization`/`digest` → local, `research` → cloud_standard, `complex_technical` → cloud_advanced
- **Budget**: Daily spend tracked in Supabase. Over budget → downgrade cloud to local
- **Override**: `--model local|cloud` on CLI bypasses the router entirely
- `route_and_get_llm()` returns `(llm_instance, tier_name, cost_per_call)`

## Known Patterns and Issues

- **LLM narrates tool calls**: Local models sometimes describe what they'd do instead of producing structured tool_call JSON. The `FALLBACK_DISPATCH` in `agent/act.py` handles this by keyword-matching the LLM's text output to dispatch tools.
- **Keyword fallback in classifier**: Classification is entirely keyword-based (no LLM). If no keywords match, defaults to "research" with research-focused tools.
- **Subprocess fallback is opt-in**: `code_executor` falls back to subprocess only when `--allow-subprocess` CLI flag is set or `ALLOW_SUBPROCESS=true` is in `.env`. Without this, Docker unavailability returns an error. The subprocess fallback uses AST-based safety checks but these are bypassable — Docker is the primary security boundary.
- **Redundancy detection**: `reflect_node` stops the loop if the same tool is called 3+ times or the same result appears twice.
- **Two-stage tool promotion (by design)**: Metatool-created tools only go live in the daemon/CLI after human approval (stage 1). To promote a tool to the MCP server for cloud clients (claude.ai, Desktop, mobile), Claude Code adds it as an `@mcp.tool()` in `server.py` and redeploys to Railway (stage 2). This is intentional — no tool reaches cloud clients without two gates.
- **Digest dedup on ingest**: `queue_for_review()` checks existing `_digest_review` items by URL and title before inserting, preventing reviewed items from reappearing on subsequent pipeline runs.
- **Ollama auto-start**: `check_services()` in `main.py` attempts to start Ollama via `subprocess.Popen(["ollama", "serve"])` if it's not reachable. Waits 3 seconds and retries. Falls back to a warning if still unreachable. The check always runs (not gated on `agent_model`) since the digest pipeline uses `get_llm("local")` directly.

## Digest Pipeline

`agent/digest.py` is a standalone orchestrator that chains existing tools to produce a daily research digest. It does NOT use the LangGraph graph — it calls tools and LLMs directly.

**Flow**: `run_digest(frequency)` → load sources from `config/digest_sources.yaml` → for each source, fetch via `url_reader` or `arxiv_search` → LLM (local 32B) extracts items → **embedding pre-filter** compares each item against SCMS project memories via `get_embeddings_batch()` + `search_memories_by_embedding()`, filters items below per-source `similarity_threshold` → LLM scores only items that passed pre-filter → build markdown digest → save to `~/Documents/cairn/digests/` → queue high-relevance items in task_queue (project=`_digest_review`) with both relevance and embedding scores for human review → macOS notification.

**Embedding Pre-Filter**: Each item's title+summary is embedded and compared against memories in the source's `relevance_projects`. The max cosine similarity across all projects determines if the item passes. Cold start bypass: if SCMS has no memories for target projects, all items pass through. Per-source `similarity_threshold` in `digest_sources.yaml` overrides the global default (0.3). As the user stores more memories, filtering automatically improves — a feedback loop.

**Few-Shot Calibration**: `_build_few_shot_context()` pulls recent approved/rejected item titles from `task_queue` and injects them into the LLM scoring prompt as labeled examples. Requires 3+ approved items; falls back to the generic prompt otherwise. Improves scoring calibration as the user reviews more items.

**Dedup on Ingest**: `queue_for_review()` fetches existing `_digest_review` items and skips duplicates by URL (primary) or title (fallback), preventing previously reviewed items from reappearing.

**Evaluation Pipeline**: `agent/evaluation.py` mines approval/rejection history from `task_queue` to compute metrics: approval rates by relevance/embedding score buckets, per-source breakdown, F1-optimal thresholds, and weekly trends. Generates a markdown report saved to the digests directory.

**CLI**: `--digest` (run manually), `--review-digest` (approve/reject items into SCMS), `--digest-status` (show recent runs), `--digest-eval` (run evaluation report).

**Daemon**: Tasks with "digest" keywords bypass the graph and call `run_digest()` directly. Set up via `--queue "Run daily digest" --recurring "0 6 * * *"`.

**Cost**: ~$0/day (local model for summarization, ~$0.001/day for embedding pre-filter via OpenAI `text-embedding-3-small`).

## Config System

`config/settings.py` uses Pydantic `BaseSettings` loading from `.env`. Key settings:
- `supabase_url`, `supabase_key` — database connection
- `ollama_base_url`, `ollama_model` — local LLM
- `anthropic_api_key`, `agent_model` — cloud LLM
- `max_iterations` (10), `max_tool_calls_per_step` (5)
- `allowed_directories` — file access whitelist
- `docker_host`, `sandbox_image` — Docker sandbox
- `daemon_poll_interval` (30s), `daily_budget_usd` (5.00)
- `mcp_base_url` — public Railway URL; enables OAuth when set (MCP server)
- `mcp_host` (0.0.0.0), `mcp_port` (8000) — MCP server bind address

## Database (Supabase + pgvector)

5 tables: `projects`, `memories` (with vector(1536) embedding), `tool_registry`, `decision_log`, `task_queue`. Semantic search via `match_memories()` RPC using cosine distance on HNSW index. Embeddings generated via OpenAI `text-embedding-3-small`.

## MCP Server

`mcp_server/server.py` exposes SCMS as 16 tools over Streamable HTTP using FastMCP:

- `scms_search`, `scms_store` — semantic memory search and storage
- `get_project_context`, `list_projects` — project browsing
- `create_project`, `update_project`, `archive_project` — project CRUD
- `queue_task`, `check_queue`, `get_task_result` — task queue management
- `get_decisions`, `log_decision` — decision log access
- `agent_status` — queue counts and daily spend
- `review_digest`, `digest_status` — digest pipeline review and monitoring
- `digest_eval` — digest evaluation metrics report (approval rates, threshold analysis)

**Auth**: OAuth 2.1 via `InMemoryOAuthProvider` with Dynamic Client Registration (DCR) and PKCE. Enabled when `MCP_BASE_URL` is set; no auth for local dev. Clients re-authenticate after Railway deploys (handled automatically by the OAuth flow).

**Deployment**: Railway via `railway up`. Build uses `Dockerfile.mcp` (python:3.12-slim + uv). Health check at `/health`.

**Annotations**: Tools declare MCP `ToolAnnotations` so clients like claude.ai don't gate calls behind approval prompts. Read-only tools (search, list, status) use `readOnlyHint: True`; write tools (store, create, update, archive, queue, log) use `destructiveHint: False`.

**Clients**: claude.ai (connector), Claude Desktop (mcp-remote), Claude Code (`--transport http`), mobile (via claude.ai).

## Dependencies

LangGraph + LangChain ecosystem for agent orchestration. Supabase for persistence. Docker SDK for sandboxing. FastMCP for MCP server + OAuth. DuckDuckGo, trafilatura, arxiv for data sources. Rich for terminal UI. Croniter for recurring task scheduling.

## Testing

```bash
uv run pytest                              # Run all tests
uv run pytest tests/test_classifier.py -v  # Run specific test
```

Test files:
- `tests/test_classifier.py` — classify_task keyword matching + detect_project fuzzy matching
- `tests/test_plan_parser.py` — parse_plan_steps: numbered steps, action filtering, caps, early exit
- `tests/test_model_router.py` — classify_complexity routing rules + route_and_get_llm override/budget
- `tests/test_mcp_server.py` — MCP tool functions with mocked SCMSClient
- `tests/test_project_crud.py` — SCMSClient CRUD methods, MCP project tools, classifier routing
- `tests/test_graph_integration.py` — Full classify→plan→act→reflect loop through build_graph()
- `tests/test_metatool_loading.py` — Custom tool @tool decorator wrapping on load
- `tests/test_digest_prefilter.py` — Embedding pre-filter: threshold, cold start, per-source override
- `tests/test_digest_dedup.py` — Queue dedup by URL/title, extract helpers, cancelled completed_at
- `tests/test_digest_fewshot.py` — Few-shot calibration: sufficient history, fallback, limits
- `tests/test_evaluation.py` — Eval pipeline: parsing, metrics, buckets, thresholds, report generation

Tests use mocked SCMS client — no Supabase, Docker, or API keys needed. Dev dependencies: `uv sync --group dev`.
