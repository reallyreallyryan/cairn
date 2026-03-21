# cairn

A self-extending AI agent with persistent memory, sandboxed tool creation, budget-aware model routing, and cloud access via MCP.

Built with LangGraph, Supabase/pgvector, Docker, and FastMCP. Developed across 6 disciplined sprints, each adding a distinct capability layer.

---

## What Makes This Different

Most LangGraph agent repos are single-feature demos — a memory example here, a tool-calling example there. cairn is an integrated system where every piece works together, designed for individual developers who want to build and run a personal AI agent without enterprise infrastructure.

- **Self-Extending Metatool System with Human Approval** — The agent can write its own tools, test them in a Docker sandbox, register them in a database as "pending," and require explicit human review and approval via CLI before they go live. Self-extending agents exist (see [Related Projects](#related-projects)), but most auto-promote new capabilities. cairn's full pipeline — create → sandbox test → DB registration → human review → approval → dynamic import — prioritizes safety over convenience.

- **4-Tier Model Routing with Budget Caps** — YAML-driven rules route tasks to the cheapest capable model (Qwen 3 8B → Qwen 3 32B → Claude Sonnet → Claude Extended). Daily spend tracking auto-downgrades to local models when you hit your budget. No LiteLLM dependency — just a simple, readable routing config.

- **Daily Research Digest Pipeline** — A recurring daemon task scrapes configurable news sources, summarizes with a local 32B model, scores relevance against your projects, and queues items for human review. Approved items become permanent memories. Runs for ~$0.03/month.

- **MCP Server with OAuth 2.1** — Your agent's memory and task queue are accessible from claude.ai, Claude Desktop, Claude Code, and mobile via a Railway-deployed MCP server with full OAuth 2.1 (DCR + PKCE). One of the few Python FastMCP + OAuth reference implementations available.

- **Persistent Memory (SCMS)** — Supabase + pgvector with 1536-dimensional embeddings and HNSW cosine search. The agent remembers across sessions — projects, decisions, learnings, and context. Not a demo — this is the actual persistence layer the whole system runs on.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (main.py)                     │
│  Modes: single task, interactive REPL, daemon, digest    │
└──────────────┬──────────────────────────────┬────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────┐
│   LangGraph StateGraph    │    │    Daemon (daemon.py)    │
│                           │    │  APScheduler poll loop   │
│  START                    │    │  Recurring cron tasks    │
│    → CLASSIFY (no LLM)    │    │  Daily digest pipeline   │
│    → PLAN    (LLM)        │    └─────────────────────────┘
│    → ACT     (LLM+tools)  │
│    → REFLECT (LLM)        │
│    ...or END              │
└──────────┬────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                    Tool Layer (13+ tools)                  │
│  Data:  web_search, url_reader, arxiv_search,             │
│         github_search                                     │
│  Files: file_reader, file_writer, note_taker              │
│  Code:  code_executor (Docker sandbox)                    │
│  SCMS:  scms_search, scms_store                           │
│  Meta:  create_tool, test_tool, list_pending_tools        │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────┐  ┌──────────────────┐  ┌──────────────┐
│  Supabase/pgvector  │  │  Docker Sandbox   │  │ Model Router  │
│  5 tables + RPC     │  │  256MB, no net    │  │ YAML rules    │
│  1536-dim embeddings│  │  60s timeout      │  │ 4 tiers       │
└────────────────────┘  └──────────────────┘  └──────────────┘

               ┌──────────────────────────────────────────┐
               │  MCP Server (Railway)                     │
               │  FastMCP + OAuth 2.1 (DCR + PKCE)        │
               │  10 tools · claude.ai / Desktop / mobile  │
               └──────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) for local LLM inference
- [Docker](https://www.docker.com/) (optional, for sandboxed code execution)
- A [Supabase](https://supabase.com/) project (free tier works)

### Setup

```bash
git clone https://github.com/reallyreallyryan/cairn.git
cd cairn

# Install dependencies
uv sync

# Pull local models
ollama pull qwen3:32b
ollama pull qwen3:8b

# Set up Supabase:
# 1. Create a project at supabase.com
# 2. Enable the pgvector extension (Database > Extensions)
# 3. Run the migrations in order: scms/migrations/001_initial.sql through 004b
# 4. Copy your Project URL + anon key

# Configure
cp .env.example .env
# Edit .env with your Supabase URL, API keys, etc.

# Start Ollama
ollama serve

# Run your first task
uv run python main.py "What projects am I working on?"
```

## Usage

```bash
# Single task
python main.py "Search the web for LangGraph tutorials"

# Interactive mode
python main.py -i

# Use cloud model explicitly
python main.py --model cloud "Architect a REST API for task management"

# Daily research digest
python main.py --digest              # Run manually
python main.py --review-digest       # Review & approve/reject items into memory
python main.py --digest-status       # Check last run stats

# Task queue & daemon
python main.py --queue "Research MCP best practices" --priority 2
python main.py --daemon              # Background task processing

# Metatool management
python main.py --pending-tools       # List tools awaiting approval
python main.py --review-tool <id>    # Review tool code + sandbox test results
python main.py --approve-tool <id>   # Approve for production use
```

## Sprint History

cairn was built incrementally across 6 sprints. Each sprint added a distinct capability layer, and each sprint brief was handed to Claude Code for implementation.

| Sprint | Focus | What Was Added |
|--------|-------|----------------|
| 1 | Foundation | SCMS + pgvector memory, 10 tools, CLI with single task and interactive modes |
| 2 | Intelligence | Plan→Act→Reflect loop, keyword classifier, multi-step planning, decision logging |
| 3 | Security | Docker sandbox, metatool system, human approval workflow, dynamic tool loading |
| 4 | Autonomy | 3-tier model routing, budget tracking, task queue, daemon mode, notifications |
| 5a | Cloud Access | MCP server, OAuth 2.1, Railway deployment, OpenAI embedding migration |
| 5b | Digest Pipeline | Daily research digest, 4-tier routing (Qwen 3 upgrade), local 32B summarization |

## Project Structure

```
├── agent/                # LangGraph agent
│   ├── graph.py          # StateGraph: classify → plan → act → reflect
│   ├── classifier.py     # Keyword-based task classification (no LLM call)
│   ├── nodes.py          # Node implementations
│   ├── state.py          # AgentState TypedDict
│   ├── model_router.py   # Complexity → tier → budget check → LLM instance
│   ├── daemon.py         # Background task queue processor
│   ├── digest.py         # Daily research digest orchestrator
│   ├── notifications.py  # macOS + file log notifications
│   └── tools/            # 13+ tools (web, files, code, SCMS, metatool)
│       ├── web_search.py
│       ├── url_reader.py
│       ├── arxiv_search.py
│       ├── github_search.py
│       ├── file_reader.py
│       ├── file_writer.py
│       ├── note_taker.py
│       ├── code_executor.py
│       ├── scms_tools.py
│       ├── metatool.py
│       └── custom/       # Agent-created tools (after human approval)
├── mcp_server/           # FastMCP server for cloud access
│   ├── server.py         # 10 MCP tools, OAuth 2.1
│   └── config.py
├── config/               # YAML configs
│   ├── model_routing.yaml
│   ├── sandbox_policy.yaml
│   └── digest_sources.yaml
├── scms/                 # Shared Context Memory Store
│   ├── client.py         # SCMSClient — CRUD + semantic search
│   ├── embeddings.py     # OpenAI text-embedding-3-small
│   └── migrations/       # Supabase SQL migrations (001–004b)
├── sandbox/              # Docker sandbox
│   ├── Dockerfile
│   └── manager.py        # Container lifecycle, code injection, cleanup
└── main.py               # CLI entry point
```

## Design Decisions

Key choices and their tradeoffs:

- **Keyword classifier over LLM classifier** — Task classification uses deterministic keyword matching, not an LLM call. Faster, cheaper, predictable. Falls back to "research" with all tools for ambiguous tasks.
- **Supabase over SQLite** — pgvector for semantic search, cloud-accessible from MCP server, single source of truth. Requires network connectivity but enables the entire cloud access story.
- **Flat cost estimates over token tracking** — Simple $0/$0.01/$0.03 per-call tiers rather than token-level metering. Sufficient for budget guardrails. Token-level tracking deferred to future work.
- **Human approval for agent-created tools** — The metatool pipeline requires explicit CLI approval. No auto-promotion, ever. This is a deliberate safety decision.
- **Local-first model routing** — Default tier is local (free). Cloud models only used when routing rules determine the task needs them. Budget exhaustion auto-downgrades to local.

## Cost

cairn is designed to be cheap to run daily:

| Operation | Model | Cost |
|-----------|-------|------|
| Simple recall / notes | Qwen 3 8B (local) | $0.00 |
| Summarization / digest | Qwen 3 32B (local) | $0.00 |
| Research / multi-step | Claude Sonnet (cloud) | ~$0.01/task |
| Complex technical | Claude Sonnet extended | ~$0.03/task |
| Daily digest (full run) | Local + embedding | ~$0.001/day |
| **Daily budget cap** | Configurable | Default $5.00 |

The daily digest pipeline runs almost entirely on local models. The only cloud cost is embedding approved items via OpenAI text-embedding-3-small (~$0.03/month).

## Roadmap

- [ ] Improve digest relevance scoring (few-shot calibration from approval/rejection history)
- [ ] Text-to-speech morning briefing from digest output
- [ ] 24/7 daemon deployment on Railway
- [ ] Evaluation pipeline using digest approval/rejection data
- [ ] Memory deduplication and aging
- [ ] Multi-agent collaboration patterns

## Related Projects

cairn exists in a growing ecosystem of autonomous agent tools. These projects explore overlapping ideas at different scales:

- **[OpenClaw](https://github.com/openclaw/openclaw)** — Personal AI assistant with 214k+ stars. Connects to messaging platforms with self-extending skills. Different architecture (gateway vs. research agent) and auto-promotes new capabilities without human approval.
- **[NVIDIA OpenShell](https://developer.nvidia.com/blog/run-autonomous-self-evolving-agents-more-safely-with-nvidia-openshell/)** — Enterprise sandbox for self-evolving agents with policy controls. Requires DGX/RTX hardware. cairn targets the same safety-first philosophy at a scale that runs on a laptop with Ollama.
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — The state machine framework cairn is built on. cairn's Classify→Plan→Act→Reflect loop is one opinionated implementation of LangGraph's primitives.
- **[LiteLLM](https://github.com/BerriAI/litellm)** — Production LLM proxy with routing and budget management at enterprise scale. cairn's model router is a lightweight alternative for solo developers who want the same idea in a YAML file.
- **[e2b](https://github.com/e2b-dev/e2b)** — Cloud sandboxing for AI code execution. cairn uses a simpler local Docker sandbox with resource limits.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

If you build something interesting with cairn, I'd love to hear about it.

## License

MIT — see [LICENSE](LICENSE) for details.
