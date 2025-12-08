import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './i18n' // Initialize i18n before App
import App from './App.tsx'
import { AuthProvider } from './contexts/AuthContext'
import { CitationProvider } from './contexts/CitationContext'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthProvider>
      <CitationProvider>
        <App />
      </CitationProvider>
    </AuthProvider>
  </StrictMode>,
)
