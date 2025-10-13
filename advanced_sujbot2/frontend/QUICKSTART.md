# Quick Start Guide

## Installation

### Prerequisites
- Node.js 18 or higher
- npm or pnpm

### Install Dependencies

```bash
cd frontend
npm install
```

Or with pnpm:
```bash
pnpm install
```

## Development

### 1. Create Environment File

```bash
cp .env.example .env
```

Edit `.env`:
```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
VITE_APP_VERSION=1.0.0
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

### 3. Features to Test

- **Language Switcher** - Toggle between Czech and English
- **Theme Toggle** - Switch between light and dark mode
- **Document Upload** - Drag and drop or click to upload files
- **Chat Interface** - Type messages and see responses (requires backend)

## Production Build

```bash
npm run build
npm run preview
```

## Docker

### Build Image
```bash
docker build -t sujbot2-frontend .
```

### Run Container
```bash
docker run -p 80:80 sujbot2-frontend
```

Access at `http://localhost`.

## Testing

### Unit Tests
```bash
npm run test
```

### E2E Tests
```bash
npm run test:e2e
```

### Linting
```bash
npm run lint
```

### Type Checking
```bash
npm run type-check
```

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   ├── stores/         # Zustand stores
│   ├── services/       # API services
│   ├── types/          # TypeScript types
│   ├── utils/          # Utilities
│   ├── locales/        # i18n translations
│   └── App.tsx         # Main app
├── public/             # Static assets
├── e2e/                # E2E tests
└── package.json
```

## Common Tasks

### Add New Component
```bash
# Create component file
touch src/components/MyComponent.tsx
```

### Add Translation
Edit `src/locales/cs/common.json` and `src/locales/en/common.json`.

### Add New Store
```bash
touch src/stores/myStore.ts
```

### Add API Endpoint
Edit `src/services/api.ts`.

## Troubleshooting

### Port 3000 in use
```bash
vite --port 3001
```

### WebSocket not connecting
Check that backend is running at `http://localhost:8000`.

### Styles not working
```bash
npm run build
```

### TypeScript errors
```bash
npm run type-check
```

## Next Steps

1. Start backend server (see backend documentation)
2. Upload a contract document
3. Upload a law document
4. Ask questions in the chat
5. View compliance analysis results

## Resources

- [React Documentation](https://react.dev)
- [Vite Documentation](https://vitejs.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [shadcn/ui](https://ui.shadcn.com)
- [Zustand](https://github.com/pmndrs/zustand)
- [react-i18next](https://react.i18next.com)
