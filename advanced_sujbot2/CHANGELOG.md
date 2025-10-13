# Changelog

All notable changes to the SUJBOT2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2025-01-11

#### Real-time Pipeline Status Display
- **Frontend**: Added real-time pipeline status visualization in chat interface
  - Animated progress bar showing 0-100% completion
  - Step counter (e.g., "2/3") for current pipeline stage
  - Status messages with emoji icons (🔍, ✨, ⚖️, etc.)
  - Bilingual support (Czech/English) for all status messages
  - Files modified:
    - `frontend/src/types/index.ts` - Added `PipelineStatus` and `PipelineType` interfaces
    - `frontend/src/stores/chatStore.ts` - Added `updatePipelineStatus()` and `attachSources()` actions
    - `frontend/src/App.tsx` - WebSocket handler for `pipeline_status` messages
    - `frontend/src/components/BotMessage.tsx` - UI components for status display

- **Backend**: Pipeline status emission for all query types
  - **Simple Query Pipeline** (3 steps):
    - Step 1/3: Document retrieval (33% progress)
    - Step 2/3: Answer generation (66% progress)
    - Step 3/3: Finalization (100% progress)
  - **Compliance Analysis Pipeline** (5 steps):
    - Step 1/5: Initialization (20% progress)
    - Step 2/5: Legal requirements extraction (40% progress)
    - Step 3/5: Compliance analysis (60% progress)
    - Step 4/5: Report generation (80% progress)
    - Step 5/5: Report streaming (implied complete)
  - **Cross-Document Pipeline** (3 steps):
    - Step 1/3: Initialization (33% progress)
    - Step 2/3: Document matching (66% progress)
    - Step 3/3: Analysis generation (100% progress)
  - Files modified:
    - `backend/app/routers/websocket.py` - Parser for `__STATUS__{json}__STATUS__` markers
    - `backend/app/services/chat_service.py` - Status emission in all pipeline methods

#### Collapsible Context/Sources UI
- **Frontend**: Improved UX for displaying source citations
  - Collapsible "Context" button showing number of sources
  - Click to expand/collapse source list
  - Badge-style display for each source with:
    - Legal reference (e.g., "§ 1234")
    - Page number (if available)
    - Confidence score percentage
  - Smooth expand/collapse animation
  - Files modified:
    - `frontend/src/components/BotMessage.tsx` - Collapsible UI implementation

#### Internationalization
- Added i18n translations for new UI elements:
  - Czech: "Kontext", "zdroj", "Zpracovávám"
  - English: "Context", "source", "Processing"
  - Files modified:
    - `frontend/src/locales/cs/common.json`
    - `frontend/src/locales/en/common.json`

### Technical Details

#### WebSocket Status Protocol
```
Format: __STATUS__{json}__STATUS__

Example:
__STATUS__{
  "type": "pipeline_status",
  "pipeline": "simple_query",
  "stage": "retrieval",
  "stage_name": "Retrieval",
  "step": 1,
  "total_steps": 3,
  "progress": 33,
  "message": "🔍 Vyhledávám relevantní dokumenty..."
}__STATUS__
```

The WebSocket handler in `backend/app/routers/websocket.py` parses these markers and sends them as separate JSON messages to the frontend, preventing them from appearing in the response text.

#### State Management Flow
1. Backend emits status via special markers during streaming
2. WebSocket handler extracts and parses JSON
3. Frontend `App.tsx` receives `pipeline_status` message
4. `chatStore.updatePipelineStatus()` updates current message state
5. `BotMessage` component renders progress bar and status

#### Component Architecture
- **Separation of Concerns**: Pipeline status is separate from message content
- **Real-time Updates**: Status updates arrive during streaming, not after
- **Automatic Cleanup**: Pipeline status is cleared when streaming completes
- **Fallback Handling**: Shows "Processing..." if no status is available

### Performance Considerations
- Status messages are lightweight JSON (~200 bytes each)
- No impact on streaming performance
- Progress bar uses CSS transitions for smooth animation
- Collapsible context reduces visual clutter for long source lists

### Developer Notes
- All pipeline methods in `chat_service.py` now emit status
- Status messages are bilingual (Czech/English) based on language parameter
- Frontend handles missing status gracefully with fallback UI
- Pipeline status is TypeScript-typed for type safety

---

## [1.0.0] - 2025-01-10

### Added
- Initial v2 release with comprehensive RAG pipeline
- React + TypeScript frontend with ChatGPT-like interface
- FastAPI + Celery backend with async processing
- Multi-document FAISS indexing with BGE-M3 embeddings
- Hybrid retrieval (semantic + keyword + structural)
- Cross-document retrieval for contract-law comparison
- Knowledge graph for legal reference tracking
- Compliance analysis with conflict detection
- WebSocket streaming for real-time responses
- Bilingual support (Czech/English)
- Docker Compose deployment configuration
- Comprehensive technical specifications (15 specs)

### Changed
- Complete architecture redesign from v1
- Migrated from monolithic to microservices architecture
- Improved chunking strategy with § boundaries
- Enhanced retrieval with graph-aware reranking

### Fixed
- Code review issues from initial implementation
- Citation tracking and duplicate indexing bugs
- Environment variable integration and error handling
- Document analyzer warnings and errors
- Vector search functionality

---

## Previous Versions

See git history for changes before v1.0.0:
```bash
git log --oneline --before="2025-01-10"
```
