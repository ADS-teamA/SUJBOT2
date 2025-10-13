# SUJBOT2 Frontend

Modern, responsive web interface for legal compliance checking with ChatGPT-like UX.

## Features

- **ChatGPT-like Interface** - Familiar, modern chat UI
- **Dual Document Panels** - Separate areas for contracts (left) and laws (right)
- **Real-time Processing** - WebSocket chat with live streaming responses
- **Bilingual Support** - Czech ↔ English language switcher
- **Document Cards** - Rich file metadata display (pages, size, format)
- **Drag-and-drop Upload** - Easy file upload with validation
- **Dark Mode** - Full dark mode support
- **Responsive Design** - Desktop-first with mobile support

## Technology Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type-safe development
- **Vite** - Lightning-fast development server
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Accessible component library
- **Zustand** - State management
- **TanStack Query** - Data fetching and caching
- **react-i18next** - Internationalization
- **WebSocket** - Real-time communication

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`.

### Environment Variables

Create a `.env` file:

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
VITE_APP_VERSION=1.0.0
```

## Available Scripts

```bash
# Development
npm run dev          # Start dev server

# Build
npm run build        # Production build
npm run preview      # Preview production build

# Testing
npm run test         # Run unit tests
npm run test:ui      # Run tests with UI
npm run test:e2e     # Run E2E tests

# Code Quality
npm run lint         # Lint code
npm run type-check   # Type check
```

## Project Structure

```
frontend/
├── src/
│   ├── components/        # React components
│   │   ├── ui/           # shadcn/ui base components
│   │   ├── TopBar.tsx    # Top navigation bar
│   │   ├── DocumentPanel.tsx
│   │   ├── ChatArea.tsx
│   │   ├── DocumentCard.tsx
│   │   ├── UploadButton.tsx
│   │   ├── ChatInput.tsx
│   │   ├── BotMessage.tsx
│   │   ├── UserMessage.tsx
│   │   ├── LanguageSwitcher.tsx
│   │   └── ThemeToggle.tsx
│   ├── stores/           # Zustand state stores
│   │   ├── documentStore.ts
│   │   ├── chatStore.ts
│   │   └── uiStore.ts
│   ├── services/         # API services
│   │   ├── api.ts
│   │   ├── axios.ts
│   │   └── websocket.ts
│   ├── types/            # TypeScript types
│   │   └── index.ts
│   ├── utils/            # Utility functions
│   │   ├── cn.ts
│   │   ├── file.ts
│   │   └── date.ts
│   ├── locales/          # i18n translations
│   │   ├── cs/
│   │   │   └── common.json
│   │   └── en/
│   │       └── common.json
│   ├── App.tsx           # Main app component
│   ├── main.tsx          # Entry point
│   ├── i18n.ts           # i18n configuration
│   └── index.css         # Global styles
├── public/               # Static assets
├── Dockerfile            # Docker configuration
├── nginx.conf            # Nginx configuration
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Component Architecture

### Layout
- `TopBar` - Application header with logo, language switcher, theme toggle
- `DocumentPanel` - Left/right panels for contracts and laws
- `ChatArea` - Central chat interface with messages and input

### Document Management
- `UploadButton` - Drag-and-drop file upload
- `DocumentCard` - Document metadata card with status and actions

### Chat
- `ChatInput` - Message input with auto-resize
- `UserMessage` - User message display
- `BotMessage` - Bot message with markdown, sources, citations

### UI Controls
- `LanguageSwitcher` - Czech/English toggle
- `ThemeToggle` - Light/dark mode toggle

## State Management

### Document Store
- Manages contracts and laws
- Handles file upload with progress tracking
- Document status updates (uploading → processing → indexed)

### Chat Store
- Message history
- Streaming message updates
- WebSocket message handling

### UI Store
- Language preference (persisted)
- Theme preference (persisted)
- Sidebar state

## API Integration

### REST API
```typescript
// Upload document
uploadDocument(file, 'contract', onProgress);

// Get document status
getDocumentStatus(documentId);

// Delete document
deleteDocument(documentId);

// Start compliance check
startComplianceCheck(contractId, lawIds);

// Get compliance report
getComplianceReport(taskId);
```

### WebSocket
```typescript
// Connect to WebSocket
chatWebSocket.connect('ws://localhost:8000/ws');

// Send message
chatWebSocket.sendMessage('What are the payment terms?');

// Handle incoming messages
chatWebSocket.onMessage((data) => {
  if (data.type === 'chat_chunk') {
    updateStreamingMessage(data.content);
  }
});
```

## Supported File Formats

- **PDF** (.pdf) - Primary format
- **DOCX** (.docx) - Microsoft Word
- **TXT** (.txt) - Plain text
- **Markdown** (.md)
- **ODT** (.odt) - OpenDocument
- **RTF** (.rtf) - Rich Text Format

Maximum file size: 500 MB

## Internationalization

The app supports Czech and English. Translations are in `src/locales/`.

### Adding Translations

1. Add key to `locales/cs/common.json`
2. Add English translation to `locales/en/common.json`
3. Use in components: `const { t } = useTranslation(); t('your.key')`

## Docker Deployment

```bash
# Build image
docker build -t sujbot2-frontend .

# Run container
docker run -p 80:80 sujbot2-frontend
```

The Dockerfile uses multi-stage build:
1. **Builder stage** - Installs dependencies and builds app
2. **Production stage** - Serves with nginx

## Development Tips

### Hot Module Replacement
Vite provides instant HMR. Changes appear immediately without full reload.

### Type Checking
TypeScript is configured strictly. Run `npm run type-check` to verify types.

### Tailwind IntelliSense
Install the Tailwind CSS IntelliSense VSCode extension for autocomplete.

### Component Testing
Use Vitest for unit tests:

```typescript
import { render, screen } from '@testing-library/react';
import { DocumentCard } from './DocumentCard';

test('renders document name', () => {
  render(<DocumentCard document={mockDoc} />);
  expect(screen.getByText('contract.pdf')).toBeInTheDocument();
});
```

## Performance Optimization

- **Code splitting** - Lazy load heavy components
- **Tree shaking** - Vite removes unused code
- **Chunk optimization** - Manual chunks for vendors
- **Gzip compression** - Nginx compresses assets

## Troubleshooting

### Port 3000 already in use
```bash
# Use different port
vite --port 3001
```

### WebSocket connection failed
Check that backend is running and `VITE_WS_URL` is correct.

### Tailwind styles not applying
Run `npm run build` to regenerate Tailwind classes.

### TypeScript errors
Run `npm run type-check` to see all type errors.

## License

MIT

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Support

For issues and questions, please open a GitHub issue.
