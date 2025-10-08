# 13. Frontend Architecture Specification

## 1. Purpose

**Objective**: Design and implement a modern, responsive web interface for legal compliance checking with ChatGPT-like UX, supporting document upload, real-time chat, and bilingual operation (Czech/English).

**Why Frontend?**
- Make legal compliance checking accessible to non-technical users
- Provide intuitive document management (drag-and-drop, cards)
- Real-time feedback during analysis
- Visual comparison of contracts vs. laws
- Actionable compliance reports with interactive visualization

**Key Capabilities**:
1. **ChatGPT-like Interface** - Familiar, modern chat UI
2. **Dual Document Panels** - Separate areas for contracts (left) and laws (right)
3. **Real-time Processing** - Async file upload with live progress
4. **Bilingual Support** - Czech ↔ English language switcher
5. **Document Cards** - Rich file metadata display (pages, size, format)
6. **Responsive Design** - Desktop-first with mobile support
7. **Interactive Reports** - Clickable compliance issues, severity visualization

---

## 2. UI Layout & Architecture

### 2.1 Overall Layout

```
┌────────────────────────────────────────────────────────────────┐
│                         TOP BAR                                │
│  Logo | Language: CZ/EN | User Menu | Theme Toggle            │
└────────────────────────────────────────────────────────────────┘
┌─────────────────┬──────────────────────┬─────────────────────┐
│  LEFT PANEL     │   MAIN CHAT AREA     │   RIGHT PANEL       │
│  (Contracts)    │                      │   (Laws)            │
│                 │                      │                     │
│ ┌─────────────┐ │  ┌────────────────┐ │ ┌─────────────────┐ │
│ │ Upload Btn  │ │  │                │ │ │ Upload Button   │ │
│ └─────────────┘ │  │                │ │ └─────────────────┘ │
│                 │  │  Chat Messages │ │                     │
│ ┌─────────────┐ │  │                │ │ ┌─────────────────┐ │
│ │ File Card 1 │ │  │     ▼ ▼ ▼      │ │ │  Law Card 1     │ │
│ │ contract.pdf│ │  │                │ │ │  zakon_89.pdf   │ │
│ │ 1234 pages  │ │  │                │ │ │  456 pages      │ │
│ │ ✅ Indexed  │ │  │                │ │ │  ✅ Indexed     │ │
│ └─────────────┘ │  │                │ │ └─────────────────┘ │
│                 │  │                │ │                     │
│ ┌─────────────┐ │  └────────────────┘ │ ┌─────────────────┐ │
│ │ File Card 2 │ │                      │ │  Law Card 2     │ │
│ └─────────────┘ │  ┌────────────────┐ │ └─────────────────┘ │
│                 │  │ Input Box      │ │                     │
│                 │  │ [Type here...] │ │                     │
│                 │  │      [Send]    │ │                     │
│                 │  └────────────────┘ │                     │
└─────────────────┴──────────────────────┴─────────────────────┘
```

### 2.2 Responsive Breakpoints

```css
/* Desktop (default) */
@media (min-width: 1280px) {
  .layout {
    grid-template-columns: 300px 1fr 300px; /* Sidebars + chat */
  }
}

/* Tablet */
@media (min-width: 768px) and (max-width: 1279px) {
  .layout {
    grid-template-columns: 250px 1fr 250px;
  }
}

/* Mobile */
@media (max-width: 767px) {
  .layout {
    grid-template-rows: auto 1fr; /* Stack vertically */
    /* Tabs for Contracts/Laws */
  }
}
```

---

## 3. Technology Stack

### 3.1 Core Framework: **React 18+**

**Why React?**
- Large ecosystem, excellent TypeScript support
- React Server Components for performance
- Rich component libraries (Chakra UI, shadcn/ui)
- Easy state management (Zustand, React Query)

**Alternatives Considered**:
- **Vue 3**: Simpler, but smaller ecosystem
- **Svelte**: Fastest, but less mature for large apps
- **Next.js**: Better for SSR, but overkill for SPA

**Decision**: React 18 with Vite for fast dev experience

### 3.2 UI Component Library: **shadcn/ui**

**Why shadcn/ui?**
- Radix UI primitives (accessibility built-in)
- Tailwind CSS for styling
- Copy-paste components (no npm bloat)
- Modern design system
- Excellent TypeScript support

**Components to Use**:
- `Button`, `Card`, `Dialog`, `Dropdown`, `Tabs`
- `Progress`, `Badge`, `Avatar`, `Tooltip`
- `Input`, `Textarea`, `Select`, `Switch`

### 3.3 State Management: **Zustand**

**Why Zustand?**
- Minimal boilerplate (vs Redux)
- Built-in TypeScript support
- Simple API for global state
- DevTools integration

**State Stores**:
```typescript
// Document store
interface DocumentStore {
  contracts: Document[];
  laws: Document[];
  uploadDocument: (file: File, type: 'contract' | 'law') => Promise<void>;
  removeDocument: (id: string) => void;
}

// Chat store
interface ChatStore {
  messages: Message[];
  isTyping: boolean;
  sendMessage: (text: string) => Promise<void>;
  streamResponse: (responseStream: AsyncIterator<string>) => void;
}

// UI store
interface UIStore {
  language: 'cs' | 'en';
  theme: 'light' | 'dark';
  sidebarCollapsed: boolean;
  toggleLanguage: () => void;
  toggleTheme: () => void;
}
```

### 3.4 Data Fetching: **TanStack Query (React Query)**

**Why React Query?**
- Automatic caching and refetching
- Optimistic updates
- Real-time sync with server
- Built-in loading/error states

**Query Keys**:
```typescript
const queryKeys = {
  documents: ['documents'],
  document: (id: string) => ['document', id],
  complianceReport: (contractId: string, lawIds: string[]) =>
    ['compliance', contractId, ...lawIds],
  chatHistory: ['chat', 'history']
};
```

### 3.5 Real-time Communication: **WebSocket**

**Implementation**: Native WebSocket API with auto-reconnect

```typescript
class ChatWebSocket {
  private ws: WebSocket;
  private reconnectTimeout: number = 1000;

  connect(url: string): void {
    this.ws = new WebSocket(url);
    this.ws.onmessage = this.handleMessage;
    this.ws.onclose = this.handleDisconnect;
  }

  sendMessage(message: string): void {
    this.ws.send(JSON.stringify({ type: 'chat', content: message }));
  }

  private handleDisconnect = (): void => {
    setTimeout(() => this.connect(this.url), this.reconnectTimeout);
    this.reconnectTimeout = Math.min(this.reconnectTimeout * 2, 30000);
  };
}
```

### 3.6 Internationalization: **react-i18next**

**Why i18next?**
- Industry standard for React
- Dynamic language switching
- Namespace support for large apps
- Fallback language support

**Translation Structure**:
```typescript
// locales/cs/common.json
{
  "upload": {
    "contract": "Nahrát smlouvu",
    "law": "Nahrát zákon",
    "dragDrop": "Přetáhněte soubor nebo klikněte",
    "processing": "Zpracovávám..."
  },
  "chat": {
    "placeholder": "Zeptejte se na něco...",
    "send": "Odeslat",
    "thinking": "Přemýšlím..."
  }
}

// locales/en/common.json
{
  "upload": {
    "contract": "Upload Contract",
    "law": "Upload Law",
    "dragDrop": "Drag and drop or click",
    "processing": "Processing..."
  },
  "chat": {
    "placeholder": "Ask something...",
    "send": "Send",
    "thinking": "Thinking..."
  }
}
```

### 3.7 Styling: **Tailwind CSS**

**Why Tailwind?**
- Utility-first approach
- No CSS naming conflicts
- Built-in responsive design
- Excellent dark mode support

**Configuration**:
```javascript
// tailwind.config.js
module.exports = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#3b82f6',
          900: '#1e3a8a'
        }
      }
    }
  }
};
```

### 3.8 Build Tool: **Vite**

**Why Vite?**
- Lightning-fast HMR (Hot Module Replacement)
- Native ESM support
- Optimized production builds
- Excellent TypeScript support

---

## 4. Component Architecture

### 4.1 Component Hierarchy

```
App
├── Layout
│   ├── TopBar
│   │   ├── Logo
│   │   ├── LanguageSwitcher
│   │   ├── ThemeToggle
│   │   └── UserMenu
│   │
│   ├── LeftPanel (Contracts)
│   │   ├── UploadButton
│   │   ├── DocumentList
│   │   │   └── DocumentCard[]
│   │   └── PanelHeader
│   │
│   ├── MainChatArea
│   │   ├── MessageList
│   │   │   ├── UserMessage
│   │   │   ├── BotMessage
│   │   │   │   ├── MessageText
│   │   │   │   ├── SourceCitations
│   │   │   │   └── ComplianceIssues (if applicable)
│   │   │   └── TypingIndicator
│   │   │
│   │   └── ChatInput
│   │       ├── TextArea
│   │       └── SendButton
│   │
│   └── RightPanel (Laws)
│       ├── UploadButton
│       ├── DocumentList
│       │   └── DocumentCard[]
│       └── PanelHeader
│
└── Modals
    ├── ComplianceReportModal
    ├── DocumentPreviewModal
    └── SettingsModal
```

### 4.2 Key Component Specifications

#### 4.2.1 DocumentCard

```typescript
interface DocumentCardProps {
  document: Document;
  onRemove: (id: string) => void;
  onPreview: (id: string) => void;
}

interface Document {
  id: string;
  filename: string;
  filesize: number;
  format: 'pdf' | 'docx' | 'txt';
  pageCount: number;
  wordCount: number;
  uploadedAt: Date;
  status: 'uploading' | 'processing' | 'indexed' | 'error';
  progress: number; // 0-100
  errorMessage?: string;
}

const DocumentCard: React.FC<DocumentCardProps> = ({ document, onRemove, onPreview }) => {
  return (
    <Card className="p-4 hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <FileIcon format={document.format} />
        <Badge variant={getStatusVariant(document.status)}>
          {t(`status.${document.status}`)}
        </Badge>
      </div>

      {/* Filename */}
      <h3 className="font-medium truncate" title={document.filename}>
        {document.filename}
      </h3>

      {/* Metadata */}
      <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
        <div className="flex justify-between">
          <span>{document.pageCount} {t('pages')}</span>
          <span>{formatFileSize(document.filesize)}</span>
        </div>
        <div className="mt-1">
          {document.wordCount.toLocaleString()} {t('words')}
        </div>
      </div>

      {/* Progress bar (if processing) */}
      {document.status === 'processing' && (
        <Progress value={document.progress} className="mt-3" />
      )}

      {/* Error message */}
      {document.status === 'error' && (
        <Alert variant="destructive" className="mt-3">
          {document.errorMessage}
        </Alert>
      )}

      {/* Actions */}
      <div className="flex gap-2 mt-3">
        <Button size="sm" variant="outline" onClick={() => onPreview(document.id)}>
          {t('preview')}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onRemove(document.id)}>
          {t('remove')}
        </Button>
      </div>
    </Card>
  );
};
```

#### 4.2.2 UploadButton

```typescript
interface UploadButtonProps {
  type: 'contract' | 'law';
  onUpload: (files: File[]) => Promise<void>;
  accept?: string;
  multiple?: boolean;
}

const UploadButton: React.FC<UploadButtonProps> = ({
  type,
  onUpload,
  accept = '.pdf,.docx,.txt,.md,.odt,.rtf',
  multiple = true
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    await onUpload(files);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    await onUpload(files);
  };

  return (
    <div
      className={cn(
        "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors",
        isDragging ? "border-primary bg-primary/10" : "border-gray-300 hover:border-primary"
      )}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <UploadIcon className="mx-auto h-12 w-12 text-gray-400" />
      <p className="mt-2 text-sm">
        {t(`upload.${type}`)}
      </p>
      <p className="text-xs text-gray-500 mt-1">
        {t('upload.dragDrop')}
      </p>

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept={accept}
        multiple={multiple}
        onChange={handleFileChange}
      />
    </div>
  );
};
```

#### 4.2.3 ChatInput

```typescript
interface ChatInputProps {
  onSend: (message: string) => Promise<void>;
  disabled?: boolean;
  placeholder?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled = false,
  placeholder
}) => {
  const [message, setMessage] = useState('');
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = async () => {
    if (!message.trim() || isSending) return;

    setIsSending(true);
    try {
      await onSend(message.trim());
      setMessage('');
      textareaRef.current?.focus();
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [message]);

  return (
    <div className="flex gap-2 items-end">
      <Textarea
        ref={textareaRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder || t('chat.placeholder')}
        disabled={disabled || isSending}
        className="min-h-[60px] max-h-[200px] resize-none"
      />
      <Button
        onClick={handleSend}
        disabled={!message.trim() || disabled || isSending}
        size="lg"
      >
        {isSending ? <Loader2 className="animate-spin" /> : <SendIcon />}
      </Button>
    </div>
  );
};
```

#### 4.2.4 BotMessage

```typescript
interface BotMessageProps {
  message: ChatMessage;
  onCitationClick: (source: Source) => void;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  sources?: Source[];
  complianceIssues?: ComplianceIssue[];
  isStreaming?: boolean;
}

const BotMessage: React.FC<BotMessageProps> = ({ message, onCitationClick }) => {
  return (
    <div className="flex gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
      {/* Bot avatar */}
      <Avatar>
        <AvatarImage src="/bot-avatar.png" />
        <AvatarFallback>AI</AvatarFallback>
      </Avatar>

      <div className="flex-1">
        {/* Message content with markdown */}
        <ReactMarkdown
          className="prose dark:prose-invert"
          components={{
            // Custom renderers for citations
            a: ({ href, children }) => (
              <a
                href={href}
                className="text-primary underline cursor-pointer"
                onClick={(e) => {
                  e.preventDefault();
                  // Parse citation from href
                  const source = parseCitationHref(href);
                  if (source) onCitationClick(source);
                }}
              >
                {children}
              </a>
            )
          }}
        >
          {message.content}
        </ReactMarkdown>

        {/* Source citations */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {t('sources')}:
            </p>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, idx) => (
                <Badge
                  key={idx}
                  variant="secondary"
                  className="cursor-pointer"
                  onClick={() => onCitationClick(source)}
                >
                  {source.legal_reference}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Compliance issues (if any) */}
        {message.complianceIssues && message.complianceIssues.length > 0 && (
          <div className="mt-4">
            <ComplianceIssueList issues={message.complianceIssues} />
          </div>
        )}

        {/* Streaming indicator */}
        {message.isStreaming && (
          <div className="mt-2">
            <Loader2 className="animate-spin h-4 w-4 text-gray-400" />
          </div>
        )}

        {/* Timestamp */}
        <p className="text-xs text-gray-500 mt-2">
          {formatTimestamp(message.timestamp)}
        </p>
      </div>
    </div>
  );
};
```

#### 4.2.5 ComplianceIssueList

```typescript
interface ComplianceIssueListProps {
  issues: ComplianceIssue[];
}

const ComplianceIssueList: React.FC<ComplianceIssueListProps> = ({ issues }) => {
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">{t('complianceIssues')}:</p>
      {issues.map((issue) => (
        <Card key={issue.issue_id} className="p-3">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              {/* Severity badge */}
              <Badge variant={getSeverityVariant(issue.severity)}>
                {t(`severity.${issue.severity}`)}
              </Badge>

              {/* Issue description */}
              <p className="text-sm mt-2">{issue.issue_description}</p>

              {/* Contract reference */}
              <p className="text-xs text-gray-600 mt-1">
                {t('contract')}: {issue.contract_reference}
              </p>

              {/* Law requirements */}
              {issue.law_requirements.map((req, idx) => (
                <p key={idx} className="text-xs text-gray-600">
                  {t('law')}: {req.law_reference}
                </p>
              ))}
            </div>

            {/* Expand button */}
            <Button size="sm" variant="ghost">
              <ChevronDown />
            </Button>
          </div>

          {/* Recommendations (collapsible) */}
          <Collapsible>
            <div className="mt-3 pt-3 border-t">
              <p className="text-sm font-medium">{t('recommendations')}:</p>
              <ul className="list-disc list-inside text-sm text-gray-700 mt-1">
                {issue.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          </Collapsible>
        </Card>
      ))}
    </div>
  );
};
```

#### 4.2.6 LanguageSwitcher

```typescript
const LanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();
  const [language, setLanguage] = useState<'cs' | 'en'>(
    i18n.language as 'cs' | 'en'
  );

  const toggleLanguage = () => {
    const newLang = language === 'cs' ? 'en' : 'cs';
    setLanguage(newLang);
    i18n.changeLanguage(newLang);
  };

  return (
    <div className="flex items-center gap-2">
      <span className={cn("text-sm", language === 'cs' && "font-bold")}>
        CZ
      </span>
      <Switch
        checked={language === 'en'}
        onCheckedChange={toggleLanguage}
      />
      <span className={cn("text-sm", language === 'en' && "font-bold")}>
        EN
      </span>
    </div>
  );
};
```

---

## 5. State Management Architecture

### 5.1 Document Store

```typescript
import create from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface DocumentStore {
  contracts: Document[];
  laws: Document[];

  uploadDocument: (file: File, type: 'contract' | 'law') => Promise<void>;
  removeDocument: (id: string, type: 'contract' | 'law') => void;
  updateDocumentStatus: (id: string, status: DocumentStatus, progress?: number) => void;

  getDocumentById: (id: string) => Document | undefined;
  getDocumentsByType: (type: 'contract' | 'law') => Document[];
}

export const useDocumentStore = create<DocumentStore>()(
  devtools(
    persist(
      (set, get) => ({
        contracts: [],
        laws: [],

        uploadDocument: async (file, type) => {
          const documentId = generateId();
          const newDoc: Document = {
            id: documentId,
            filename: file.name,
            filesize: file.size,
            format: getFileFormat(file.name),
            pageCount: 0,
            wordCount: 0,
            uploadedAt: new Date(),
            status: 'uploading',
            progress: 0
          };

          // Add to store immediately
          set((state) => ({
            [type === 'contract' ? 'contracts' : 'laws']: [
              ...state[type === 'contract' ? 'contracts' : 'laws'],
              newDoc
            ]
          }));

          try {
            // Upload file
            await uploadFileToServer(file, documentId, type, (progress) => {
              get().updateDocumentStatus(documentId, 'uploading', progress);
            });

            // File uploaded, now processing
            get().updateDocumentStatus(documentId, 'processing', 0);

            // Poll for indexing completion
            await pollIndexingStatus(documentId, (progress, metadata) => {
              get().updateDocumentStatus(documentId, 'processing', progress);

              // Update metadata when available
              if (metadata) {
                set((state) => {
                  const docs = type === 'contract' ? state.contracts : state.laws;
                  return {
                    [type === 'contract' ? 'contracts' : 'laws']: docs.map(d =>
                      d.id === documentId
                        ? { ...d, pageCount: metadata.pageCount, wordCount: metadata.wordCount }
                        : d
                    )
                  };
                });
              }
            });

            // Indexed successfully
            get().updateDocumentStatus(documentId, 'indexed', 100);

          } catch (error) {
            get().updateDocumentStatus(documentId, 'error');
            set((state) => {
              const docs = type === 'contract' ? state.contracts : state.laws;
              return {
                [type === 'contract' ? 'contracts' : 'laws']: docs.map(d =>
                  d.id === documentId
                    ? { ...d, errorMessage: error.message }
                    : d
                )
              };
            });
          }
        },

        removeDocument: (id, type) => {
          set((state) => ({
            [type === 'contract' ? 'contracts' : 'laws']:
              state[type === 'contract' ? 'contracts' : 'laws'].filter(d => d.id !== id)
          }));

          // Delete from server
          deleteDocumentFromServer(id);
        },

        updateDocumentStatus: (id, status, progress) => {
          set((state) => {
            const updateDocs = (docs: Document[]) =>
              docs.map(d => d.id === id ? { ...d, status, progress } : d);

            return {
              contracts: updateDocs(state.contracts),
              laws: updateDocs(state.laws)
            };
          });
        },

        getDocumentById: (id) => {
          const state = get();
          return [...state.contracts, ...state.laws].find(d => d.id === id);
        },

        getDocumentsByType: (type) => {
          const state = get();
          return type === 'contract' ? state.contracts : state.laws;
        }
      }),
      {
        name: 'document-storage',
        partialize: (state) => ({
          contracts: state.contracts,
          laws: state.laws
        })
      }
    )
  )
);
```

### 5.2 Chat Store

```typescript
interface ChatStore {
  messages: ChatMessage[];
  isTyping: boolean;
  currentStreamingMessageId: string | null;

  sendMessage: (text: string) => Promise<void>;
  addMessage: (message: ChatMessage) => void;
  updateStreamingMessage: (chunk: string) => void;
  finishStreaming: () => void;
  clearChat: () => void;
}

export const useChatStore = create<ChatStore>()(
  devtools((set, get) => ({
    messages: [],
    isTyping: false,
    currentStreamingMessageId: null,

    sendMessage: async (text) => {
      const userMessage: ChatMessage = {
        id: generateId(),
        type: 'user',
        content: text,
        timestamp: new Date()
      };

      // Add user message
      set((state) => ({
        messages: [...state.messages, userMessage]
      }));

      // Start streaming bot response
      const botMessageId = generateId();
      const botMessage: ChatMessage = {
        id: botMessageId,
        type: 'bot',
        content: '',
        timestamp: new Date(),
        isStreaming: true
      };

      set((state) => ({
        messages: [...state.messages, botMessage],
        isTyping: true,
        currentStreamingMessageId: botMessageId
      }));

      try {
        // Send to WebSocket
        const ws = getWebSocketInstance();
        ws.sendMessage(text);

        // Response will come via WebSocket onMessage handler
      } catch (error) {
        // Handle error
        set((state) => ({
          messages: state.messages.map(m =>
            m.id === botMessageId
              ? { ...m, content: 'Error: ' + error.message, isStreaming: false }
              : m
          ),
          isTyping: false,
          currentStreamingMessageId: null
        }));
      }
    },

    updateStreamingMessage: (chunk) => {
      const { currentStreamingMessageId } = get();
      if (!currentStreamingMessageId) return;

      set((state) => ({
        messages: state.messages.map(m =>
          m.id === currentStreamingMessageId
            ? { ...m, content: m.content + chunk }
            : m
        )
      }));
    },

    finishStreaming: () => {
      const { currentStreamingMessageId } = get();
      if (!currentStreamingMessageId) return;

      set((state) => ({
        messages: state.messages.map(m =>
          m.id === currentStreamingMessageId
            ? { ...m, isStreaming: false }
            : m
        ),
        isTyping: false,
        currentStreamingMessageId: null
      }));
    },

    addMessage: (message) => {
      set((state) => ({
        messages: [...state.messages, message]
      }));
    },

    clearChat: () => {
      set({ messages: [] });
    }
  }))
);
```

---

## 6. API Integration

### 6.1 HTTP Client (Axios)

```typescript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor (add auth token)
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor (handle errors)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Redirect to login
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
```

### 6.2 API Service Functions

```typescript
// services/api.ts
import apiClient from './axios';

export const uploadDocument = async (
  file: File,
  documentType: 'contract' | 'law',
  onProgress?: (progress: number) => void
): Promise<{ document_id: string }> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('document_type', documentType);

  const response = await apiClient.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = (progressEvent.loaded / progressEvent.total) * 100;
        onProgress(progress);
      }
    }
  });

  return response.data;
};

export const getDocumentStatus = async (
  documentId: string
): Promise<DocumentStatus> => {
  const response = await apiClient.get(`/documents/${documentId}/status`);
  return response.data;
};

export const deleteDocument = async (documentId: string): Promise<void> => {
  await apiClient.delete(`/documents/${documentId}`);
};

export const startComplianceCheck = async (
  contractId: string,
  lawIds: string[]
): Promise<{ task_id: string }> => {
  const response = await apiClient.post('/compliance/check', {
    contract_document_id: contractId,
    law_document_ids: lawIds,
    mode: 'exhaustive'
  });
  return response.data;
};

export const getComplianceReport = async (
  taskId: string
): Promise<ComplianceReport> => {
  const response = await apiClient.get(`/compliance/reports/${taskId}`);
  return response.data;
};
```

### 6.3 WebSocket Integration

```typescript
// services/websocket.ts
class ChatWebSocketService {
  private ws: WebSocket | null = null;
  private messageHandlers: Array<(message: any) => void> = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect(url: string): void {
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.handleReconnect(url);
    };
  }

  private handleReconnect(url: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(() => this.connect(url), delay);
    }
  }

  sendMessage(message: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'chat_message',
        content: message
      }));
    }
  }

  onMessage(handler: (message: any) => void): void {
    this.messageHandlers.push(handler);
  }

  disconnect(): void {
    this.ws?.close();
  }
}

export const chatWebSocket = new ChatWebSocketService();
```

---

## 7. Supported File Formats

### 7.1 Document Formats

**Primary Formats** (full support):
- **PDF** (.pdf) - Primary legal document format
- **DOCX** (.docx) - Microsoft Word documents
- **TXT** (.txt) - Plain text
- **Markdown** (.md) - Markdown documents

**Additional Formats** (best-effort support):
- **ODT** (.odt) - OpenDocument Text
- **RTF** (.rtf) - Rich Text Format
- **HTML** (.html, .htm) - Web pages
- **EPUB** (.epub) - E-books

### 7.2 File Validation

```typescript
const SUPPORTED_FORMATS = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
  'application/vnd.oasis.opendocument.text': ['.odt'],
  'application/rtf': ['.rtf'],
  'text/html': ['.html', '.htm'],
  'application/epub+zip': ['.epub']
};

const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500 MB

function validateFile(file: File): { valid: boolean; error?: string } {
  // Check size
  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      error: `File size exceeds maximum (${MAX_FILE_SIZE / 1024 / 1024} MB)`
    };
  }

  // Check format
  const extension = '.' + file.name.split('.').pop()?.toLowerCase();
  const isSupported = Object.values(SUPPORTED_FORMATS)
    .flat()
    .includes(extension);

  if (!isSupported) {
    return {
      valid: false,
      error: `Unsupported file format: ${extension}`
    };
  }

  return { valid: true };
}
```

---

## 8. Performance Optimization

### 8.1 Code Splitting

```typescript
// Lazy load heavy components
const ComplianceReportModal = lazy(() => import('./components/ComplianceReportModal'));
const DocumentPreviewModal = lazy(() => import('./components/DocumentPreviewModal'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));

// Usage with Suspense
<Suspense fallback={<Loader />}>
  <ComplianceReportModal />
</Suspense>
```

### 8.2 Virtualized Lists

For long document lists, use react-virtual:

```typescript
import { useVirtualizer } from '@tanstack/react-virtual';

const DocumentList: React.FC<{ documents: Document[] }> = ({ documents }) => {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: documents.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 120, // Estimated card height
    overscan: 5
  });

  return (
    <div ref={parentRef} className="h-full overflow-auto">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          position: 'relative'
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={virtualRow.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start}px)`
            }}
          >
            <DocumentCard document={documents[virtualRow.index]} />
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 8.3 Message Streaming Optimization

```typescript
// Debounce message updates to avoid excessive re-renders
import { useDebouncedCallback } from 'use-debounce';

const updateStreamingMessage = useDebouncedCallback(
  (chunk: string) => {
    chatStore.updateStreamingMessage(chunk);
  },
  50 // Update UI every 50ms max
);
```

---

## 9. Testing Strategy

### 9.1 Unit Tests (Vitest)

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentCard } from './DocumentCard';

describe('DocumentCard', () => {
  it('renders document metadata correctly', () => {
    const document: Document = {
      id: '1',
      filename: 'test.pdf',
      filesize: 1024000,
      format: 'pdf',
      pageCount: 100,
      wordCount: 5000,
      uploadedAt: new Date(),
      status: 'indexed',
      progress: 100
    };

    render(<DocumentCard document={document} onRemove={vi.fn()} onPreview={vi.fn()} />);

    expect(screen.getByText('test.pdf')).toBeInTheDocument();
    expect(screen.getByText('100 pages')).toBeInTheDocument();
    expect(screen.getByText('1.0 MB')).toBeInTheDocument();
  });

  it('shows progress bar when processing', () => {
    const document: Document = {
      ...baseDocument,
      status: 'processing',
      progress: 50
    };

    render(<DocumentCard document={document} />);

    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toHaveAttribute('aria-valuenow', '50');
  });
});
```

### 9.2 Integration Tests

```typescript
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDocumentStore } from './stores/documentStore';

describe('Document Upload Flow', () => {
  it('uploads document and updates status', async () => {
    const { result } = renderHook(() => useDocumentStore());

    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });

    await act(async () => {
      await result.current.uploadDocument(file, 'contract');
    });

    expect(result.current.contracts).toHaveLength(1);
    expect(result.current.contracts[0].filename).toBe('test.pdf');
    expect(result.current.contracts[0].status).toBe('indexed');
  });
});
```

### 9.3 E2E Tests (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test('upload and chat flow', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Upload contract
  await page.setInputFiles('input[type="file"]', 'test-contract.pdf');
  await expect(page.locator('.document-card')).toContainText('test-contract.pdf');

  // Wait for indexing
  await expect(page.locator('.document-card .badge')).toContainText('Indexed');

  // Send chat message
  await page.fill('textarea[placeholder*="Ask"]', 'What are the payment terms?');
  await page.click('button:has-text("Send")');

  // Wait for response
  await expect(page.locator('.bot-message')).toBeVisible();
  await expect(page.locator('.bot-message')).not.toContainText('Error');
});
```

---

## 10. Deployment & Build

### 10.1 Environment Variables

```bash
# .env.production
VITE_API_URL=https://api.sujbot.example.com
VITE_WS_URL=wss://api.sujbot.example.com/ws
VITE_APP_VERSION=1.0.0
```

### 10.2 Build Configuration

```javascript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
          'markdown': ['react-markdown', 'remark-gfm']
        }
      }
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
});
```

### 10.3 Docker Configuration (Frontend)

```dockerfile
# Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

COPY . .
RUN pnpm build

# Production stage
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

```nginx
# nginx.conf
server {
    listen 80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
}
```

---

## 11. Summary

Kompletní frontend pro SUJBOT2:

✅ **Moderní stack**: React 18 + TypeScript + Vite + Tailwind
✅ **ChatGPT-like UI**: Centrální chat s dvěma panely pro dokumenty
✅ **Real-time**: WebSocket pro streaming odpovědí
✅ **Bilingual**: Czech ↔ English s react-i18next
✅ **Rich document cards**: Metadata, progress, formáty
✅ **Async processing**: Nahrávání + indexování na pozadí
✅ **Responsive**: Desktop-first s mobile support
✅ **Performance**: Code splitting, virtualizace, debouncing
✅ **Testing**: Unit + Integration + E2E testy
✅ **Docker ready**: Nginx + multi-stage build

**Next Steps**:
- See [14. Backend API](14_backend_api.md) for WebSocket + REST endpoints
- See [15. Deployment](15_deployment.md) for Docker orchestration

---

**Page Count**: ~24 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅
