# SUJBOT2 Web Interface

Modern web interface for SUJBOT2 with ChatGPT/Claude-like UI featuring real-time streaming, tool visualization, and cost tracking.

## Features

✨ **Real-time Streaming** - Responses stream in real-time via Server-Sent Events (SSE)
🛠️ **Tool Visualization** - See which tools the agent uses and their results
💰 **Cost Tracking** - Monitor token usage and API costs per message
🎨 **Dark/Light Mode** - Beautiful UI with theme toggle
🤖 **Model Switching** - Switch between Claude and GPT models on the fly
💬 **Conversation History** - All conversations saved in browser LocalStorage
📤 **Export** - Export conversations in Markdown or JSON format
🎯 **Syntax Highlighting** - Code blocks with syntax highlighting
📝 **Markdown Rendering** - Full markdown support with tables and GFM

## Architecture

The web interface follows **strict separation** from the `src/` directory:
- **Zero modifications** to existing `src/` code
- **Only imports** from `src/` - no changes required
- Changes in `src/` automatically work in the web interface

### Structure

```
backend/               # FastAPI backend (Python)
├── main.py           # FastAPI app with SSE endpoints
├── agent_adapter.py  # Wrapper around src/agent/agent_core.py
├── models.py         # Pydantic schemas
└── requirements.txt  # Python dependencies

frontend/             # React + TypeScript frontend
├── src/
│   ├── components/   # React components (chat, sidebar, header)
│   ├── hooks/        # Custom hooks (useChat, useTheme)
│   ├── services/     # API client and SSE handling
│   ├── types/        # TypeScript interfaces
│   └── lib/          # LocalStorage utilities
└── package.json      # Node dependencies
```

## Quick Start

### 1. Prerequisites

- Python 3.11+ with `uv` (for backend)
- Node.js 18+ with `npm` (for frontend)
- Vector store indexed (run `run_pipeline.py` first)
- API keys configured in `.env`

### 2. Start Both Servers

```bash
# Easy way (starts both backend and frontend)
./start_web.sh
```

Or manually:

```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 3. Open Browser

Navigate to: **http://localhost:5173**

## Manual Setup

### Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt
# Or with uv: uv pip install -r requirements.txt

# Start server
python main.py
```

Backend will start on: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend will start on: **http://localhost:5173**

## Usage

### Starting a Conversation

1. Open http://localhost:5173
2. Click "New Chat" or start typing
3. Your message streams in real-time with:
   - Streaming text responses
   - Tool executions (expandable)
   - Token usage and cost

### Switching Models

1. Click the model selector in the header (Settings icon)
2. Choose from available models:
   - Claude 3.5 Sonnet v2 (most capable)
   - Claude 3.5 Haiku (fast and cheap)
   - GPT-4o (OpenAI's best)
   - GPT-4o Mini (fast OpenAI model)

### Viewing Tool Calls

When the agent uses tools (search, graph traversal, etc.):
- Tool name and status appear below the message
- Click to expand and see:
  - Input parameters
  - Execution time
  - Full results (JSON)

### Managing Conversations

- **New Chat**: Click "+ New Chat" button
- **Switch**: Click on conversation in sidebar
- **Delete**: Hover and click trash icon
- **All saved** in browser LocalStorage (persists across sessions)

### Dark/Light Mode

Click the sun/moon icon in the header to toggle themes.

## API Endpoints

### Health Check
```http
GET http://localhost:8000/health
```

Returns backend status and readiness.

### Get Models
```http
GET http://localhost:8000/models
```

Returns list of available models.

### Stream Chat (SSE)
```http
POST http://localhost:8000/chat/stream
Content-Type: application/json

{
  "message": "What is RAG?",
  "conversation_id": "optional-id",
  "model": "claude-sonnet-4-5-20250929"
}
```

Returns Server-Sent Events stream:
- `text_delta`: Streaming text chunks
- `tool_call`: Tool execution started
- `tool_result`: Tool execution completed
- `cost_update`: Token usage and cost
- `done`: Stream completed
- `error`: Error occurred

### Switch Model
```http
POST http://localhost:8000/model/switch?model=claude-3-5-haiku-20241022
```

Switches to a different model.

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **sse-starlette** - Server-Sent Events support
- **Pydantic** - Request/response validation
- **Uvicorn** - ASGI server

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Icon library
- **react-markdown** - Markdown rendering
- **rehype-highlight** - Syntax highlighting

## Development

### Backend Development

```bash
cd backend

# Run with auto-reload
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Changes to backend files automatically reload the server.

### Frontend Development

```bash
cd frontend

# Run dev server with HMR
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

Vite provides instant Hot Module Replacement (HMR) - changes appear immediately.

### Adding New Features

**Backend:**
1. Add endpoint in `backend/main.py`
2. Add schema in `backend/models.py`
3. Extend `AgentAdapter` if needed

**Frontend:**
1. Add types in `frontend/src/types/`
2. Add API method in `frontend/src/services/api.ts`
3. Create component in `frontend/src/components/`
4. Use in `App.tsx`

## Troubleshooting

### Backend won't start

```bash
# Check if vector store exists
ls -la vector_db/

# Check if API keys are set
cat .env | grep API_KEY

# Check port 8000 is free
lsof -i :8000
```

### Frontend won't start

```bash
# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install

# Check port 5173 is free
lsof -i :5173
```

### "Agent not initialized" error

The backend failed to initialize. Check:
1. Vector store exists: `vector_db/`
2. API keys are set in `.env`
3. Backend logs for detailed error

### Streaming not working

Check browser console and network tab:
1. SSE connection should show "EventStream" type
2. Events should appear in real-time
3. CORS should be enabled (should see in headers)

### LocalStorage full

Clear browser data or export conversations:
```javascript
// In browser console
localStorage.clear()
```

## Performance

- **First response**: ~200-500ms (depends on model and cache state)
- **Streaming latency**: <100ms (SSE is very fast)
- **Tool execution**: Varies by tool (100ms - 3s)
- **Memory usage**:
  - Backend: ~200MB (loaded models)
  - Frontend: ~50MB (React + components)

## Security

**Security Model:**
- ✅ Cookie-based authentication (httpOnly JWT tokens)
- ✅ Multi-user support with Argon2 password hashing
- ✅ User-owned conversations (database-backed)
- ✅ Rate limiting middleware (10 login attempts/min per IP)
- CORS enabled for localhost:5173 (development)

**For production:** Enable HTTPS, restrict CORS origins, rotate AUTH_SECRET_KEY regularly, consider implementing token blacklist in Redis for immediate revocation.

## Future Enhancements

Potential features to add:
- [ ] Export conversations (Markdown/JSON/PDF)
- [ ] Search within conversation history
- [ ] Voice input/output
- [ ] File upload for indexing
- [x] Multi-user support with auth (✅ IMPLEMENTED - PR #108)
- [ ] Real-time collaboration
- [ ] Mobile responsive design improvements
- [ ] Keyboard shortcuts
- [ ] Conversation sharing via links

## Support

For issues:
1. Check browser console for errors
2. Check backend logs
3. Verify API keys and vector store
4. See main README.md for general SUJBOT2 issues

## License

Same as main SUJBOT2 project.
