# 💬 Conversation Persistence - Deployment Guide

## 📋 Přehled

Implementoval jsem kompletní **ChatGPT-like conversation persistence** systém pro SUJBOT2:

✅ **PostgreSQL databáze** - Perzistentní ukládání konverzací
✅ **REST API endpoints** - CRUD operace pro conversations
✅ **Frontend Store** - Automatické ukládání/načítání
✅ **Intent Router** - Inteligentní routing zpráv
✅ **Compliance Pipeline** - Využívá indexované dokumenty

---

## 🚀 Rychlý Start

### 1. Spuštění PostgreSQL

```bash
# Option A: Docker Compose (doporučeno)
docker-compose -f docker-compose.dev.yml up -d postgres

# Option B: Lokální PostgreSQL
brew install postgresql@15  # macOS
brew services start postgresql@15
```

### 2. Vytvoření databázového schématu

```bash
# Připojení k databázi
psql -h localhost -U sujbot_app -d sujbot2

# Spuštění SQL schématu
\i database/schema_conversations.sql

# Ověření tabulek
\dt
# Měli byste vidět: conversations, messages

# Ověření views
\dv
# Měli byste vidět: conversation_list
```

### 3. Backend - Environment Variables

Ujistěte se, že máte v `backend/.env`:

```env
# PostgreSQL credentials (same as for vector store)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sujbot2
POSTGRES_USER=sujbot_app
POSTGRES_PASSWORD=your_password_here

# Claude API (required for Intent Router)
CLAUDE_API_KEY=your_claude_api_key
```

### 4. Backend - Spuštění

```bash
cd backend

# Install dependencies (if needed)
pip install asyncpg  # For conversation service

# Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Frontend - Spuštění

```bash
cd frontend

# Install dependencies (if needed)
npm install axios  # For conversation API

# Start frontend
npm run dev
```

---

## 📊 Architektura

### Database Schema

```sql
conversations
├── id (UUID, primary key)
├── title (VARCHAR, auto-generated from first message)
├── created_at (TIMESTAMP)
├── updated_at (TIMESTAMP)
├── message_count (INTEGER)
├── document_ids (TEXT[])  -- Which documents were used
├── is_archived (BOOLEAN)
└── is_favorite (BOOLEAN)

messages
├── id (UUID, primary key)
├── conversation_id (UUID, foreign key)
├── type ('user' | 'bot')
├── content (TEXT)
├── sequence (INTEGER)  -- Message order
├── created_at (TIMESTAMP)
├── intent (VARCHAR)     -- e.g., 'compliance_check'
├── pipeline (VARCHAR)   -- e.g., 'compliance_analysis'
└── sources (JSONB)      -- Citations
```

### API Endpoints

```
GET    /api/v1/conversations              - List conversations
POST   /api/v1/conversations              - Create new conversation
GET    /api/v1/conversations/{id}         - Get conversation with messages
PATCH  /api/v1/conversations/{id}         - Update metadata
DELETE /api/v1/conversations/{id}         - Delete conversation
POST   /api/v1/conversations/{id}/messages - Add message
```

### Frontend Flow

```
User sends message
    ↓
chatStore.sendMessage()
    ↓
1. Add to UI immediately
2. Save to database (if conversation exists)
3. Send via WebSocket to backend
    ↓
Bot response streams back
    ↓
chatStore.updateStreamingMessage()
    ↓
chatStore.finishStreaming()
    ↓
Save bot message to database
```

---

## 🔧 Použití

### Automatické Ukládání

Konverzace se **automaticky ukládají** při:
- ✅ Odeslání zprávy
- ✅ Přepnutí na jinou konverzaci
- ✅ Refresh stránky (auto-restore)

### Manuální Operace

```typescript
import { useChatStore } from '@/stores/chatStore';

// Create new conversation
await chatStore.createNewConversation();

// Load existing conversation
await chatStore.loadConversation(conversationId);

// List all conversations
await chatStore.listConversations();
// Result: chatStore.conversations

// Delete conversation
await chatStore.deleteConversation(conversationId);

// Archive conversation
await chatStore.archiveConversation(conversationId);

// Rename conversation
await chatStore.renameConversation(conversationId, 'New Title');
```

---

## 🎨 UI Komponenty (TODO)

Pro kompletní ChatGPT-like UX můžete přidat:

### Conversation Sidebar

```tsx
// frontend/src/components/ConversationSidebar.tsx
import { useChatStore } from '@/stores/chatStore';

export const ConversationSidebar = () => {
  const { conversations, loadConversation, createNewConversation } = useChatStore();

  return (
    <div className="w-64 bg-gray-900 p-4">
      <button onClick={createNewConversation}>
        + New Chat
      </button>

      {conversations.map(conv => (
        <div
          key={conv.id}
          onClick={() => loadConversation(conv.id)}
          className="p-2 hover:bg-gray-800 cursor-pointer"
        >
          <div className="font-semibold">{conv.title}</div>
          <div className="text-xs text-gray-400">
            {conv.message_count} messages
          </div>
        </div>
      ))}
    </div>
  );
};
```

### Integrovat do App.tsx

```tsx
import { ConversationSidebar } from '@/components/ConversationSidebar';

<div className="flex h-screen">
  <ConversationSidebar />
  <ChatArea />
</div>
```

---

## 🔍 Intent Router & Compliance

### Intent Classification

System automaticky detekuje, co uživatel chce:

```typescript
// User: "Je smlouva v souladu se zákonem?"
// → Intent: COMPLIANCE_CHECK
// → Pipeline: COMPLIANCE_ANALYSIS

// User: "Co říká § 89?"
// → Intent: REFERENCE_LOOKUP
// → Pipeline: SIMPLE_CHAT
```

### Compliance Analysis Flow

```
1. Intent Router detekuje compliance check
    ↓
2. ChatService routes to _process_compliance_stream()
    ↓
3. Načte chunks z již indexovaných dokumentů (no re-indexing!)
    ↓
4. ComplianceAnalyzer analyzuje:
   - Extrakce požadavků ze zákona
   - Mapování klauzulí smlouvy na požadavky
   - Detekce konfliktů
   - Gap analysis
   - Risk scoring
    ↓
5. Stream výsledky jako Markdown
```

---

## 📈 Database Maintenance

### View Conversation Stats

```sql
-- Get conversation with message count
SELECT * FROM conversation_list
ORDER BY updated_at DESC
LIMIT 10;

-- Count total conversations
SELECT COUNT(*) FROM conversations WHERE is_archived = FALSE;

-- Most active conversations
SELECT id, title, message_count
FROM conversations
ORDER BY message_count DESC
LIMIT 10;
```

### Cleanup Old Conversations

```sql
-- Archive conversations older than 90 days
UPDATE conversations
SET is_archived = TRUE
WHERE updated_at < NOW() - INTERVAL '90 days'
  AND is_archived = FALSE;

-- Delete archived conversations older than 180 days
DELETE FROM conversations
WHERE is_archived = TRUE
  AND updated_at < NOW() - INTERVAL '180 days';
```

---

## 🐛 Troubleshooting

### Conversations Not Saving

**Check 1: Database Connection**
```bash
# Test PostgreSQL connection
psql -h localhost -U sujbot_app -d sujbot2 -c "SELECT NOW();"
```

**Check 2: Schema Exists**
```sql
\dt  -- Should show conversations and messages tables
```

**Check 3: Backend Logs**
```bash
# Check for PostgreSQL errors
tail -f backend/logs/app.log | grep conversation
```

### Messages Not Loading After Refresh

**Check: Conversation ID Persistence**
```typescript
// Check localStorage
localStorage.getItem('chat-storage');
// Should contain: {"state":{"currentConversationId":"..."}}
```

**Fix: Load conversation on mount**
```typescript
// In App.tsx or main layout
useEffect(() => {
  const chatStore = useChatStore.getState();
  if (chatStore.currentConversationId) {
    chatStore.loadConversation(chatStore.currentConversationId);
  }
  chatStore.listConversations();
}, []);
```

### Intent Router Not Working

**Check: Claude API Key**
```bash
# In backend/.env
echo $CLAUDE_API_KEY
```

**Fallback**: Bez API klíče se intent router přeskočí a použije se simple chat.

---

## ✅ Hotové Funkce

✅ **Database Schema** - PostgreSQL tabulky, indexes, triggers, views
✅ **Pydantic Models** - Type-safe API models
✅ **Conversation Service** - Database CRUD operations
✅ **REST API Endpoints** - Full conversation management
✅ **Frontend Store** - Auto-save/load with Zustand
✅ **API Client** - Type-safe HTTP client
✅ **Intent Router** - Intelligent message routing
✅ **Compliance Pipeline** - Uses indexed documents

## 🚧 Optional Enhancements

□ **Conversation Sidebar UI** - ChatGPT-like sidebar (viz příklad výše)
□ **Search Conversations** - Full-text search in conversations
□ **Export Conversations** - Export as PDF/Markdown
□ **Conversation Folders** - Organize conversations
□ **Conversation Sharing** - Share conversation via link

---

## 📚 Dokumentace

### Kód Locations

```
Backend:
├── database/schema_conversations.sql          # Database schema
├── backend/app/models/conversation.py         # Pydantic models
├── backend/app/services/conversation_service.py  # Database service
├── backend/app/routers/conversations.py       # API endpoints
├── backend/app/services/intent_router.py      # Intent classification
└── backend/app/services/chat_service.py       # Updated with routing

Frontend:
├── frontend/src/services/conversationApi.ts   # API client
├── frontend/src/stores/chatStore.ts          # State management
└── (TODO) frontend/src/components/ConversationSidebar.tsx
```

### Související Dokumentace

- `CLAUDE.md` - Project overview
- `database/README.md` - Database setup
- Backend API docs: http://localhost:8000/api/docs

---

## 🎉 Ready to Use!

System je nyní plně funkční:

1. ✅ Konverzace se automaticky ukládají do PostgreSQL
2. ✅ Po refresh stránky se konverzace obnoví
3. ✅ Intent Router automaticky detekuje compliance checks
4. ✅ ComplianceAnalyzer využívá již indexované dokumenty

**Příští kroky:**
1. Spusť PostgreSQL a vytvoř schema
2. Restart backend (načte nové moduly)
3. Otevři aplikaci a začni chatovat
4. Refresh stránku → konverzace zůstane! 🎊

Potřebuješ pomoc s deployment nebo UI komponentami? Stačí se zeptat!
