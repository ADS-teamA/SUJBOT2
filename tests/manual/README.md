# Manual Tests and Diagnostics

This directory contains manual test scripts and diagnostic tools for MY_SUJBOT.

## Purpose

These scripts are **not** part of the automated test suite (`pytest`). They are used for:
- Manual verification of specific functionality
- Diagnosing issues during development
- Demonstrating correct usage patterns
- Validating production readiness

## Scripts

### Neo4j & Graph Tools

- **`test_webapp_neo4j_connection.py`** - Verifies WebApp can connect to Neo4j
  - Tests GraphAdapter initialization (same as `backend/agent_adapter.py`)
  - Validates `browse_entities` functionality
  - Run: `uv run python tests/manual/test_webapp_neo4j_connection.py`

- **`test_neo4j_agent.py`** - Tests Neo4j integration with agent
  - End-to-end test of CLI agent with Neo4j backend

- **`test_graph_adapter_methods.py`** - Tests GraphAdapter API
  - Verifies `get_outgoing_relationships()`, `get_incoming_relationships()`
  - Tests `get_chunk_by_id()` functionality

### Graph Search Tools

- **`test_browse_tool_registration.py`** - Verifies `browse_entities` tool registration
  - Tests tool is properly registered in agent
  - Validates tool parameter schema

- **`test_graph_search_usage.py`** - Demonstrates correct `graph_search` usage
  - Shows WRONG (no `entity_value`) vs CORRECT (with `entity_value`) patterns

- **`test_multi_hop_search.py`** - Tests multi-hop BFS graph traversal
  - Validates `graph_search` mode="multi_hop"

- **`test_supersession_query.py`** - Tests supersession relationship queries
  - Finds regulations with "superseded_by" relationships

## Usage

Run individual scripts:
```bash
# Test WebApp Neo4j connection
uv run python tests/manual/test_webapp_neo4j_connection.py

# Test graph adapter methods
uv run python tests/manual/test_graph_adapter_methods.py
```

**Note:** These scripts require:
- `.env` file configured with Neo4j credentials (`KG_BACKEND=neo4j`)
- Neo4j Aura instance running with indexed data
- Vector store available at `vector_db/`

## vs. Automated Tests

**Automated tests** (`tests/agent/`, `tests/graph/`):
- Run with `pytest`
- Part of CI/CD pipeline
- Use mocks/fixtures where possible
- Fast, reproducible, isolated

**Manual tests** (this directory):
- Run standalone with `uv run python`
- Use real Neo4j connections
- Require full environment setup
- For development and diagnostics

## Maintenance

These scripts are:
- ✅ Version controlled (git tracked)
- ✅ Updated with code changes
- ❌ Not run automatically in CI/CD
- ❌ Not part of test coverage metrics
