export type DocumentFormat = 'pdf' | 'docx' | 'txt' | 'md' | 'odt' | 'rtf' | 'html' | 'epub';

export type DocumentStatus = 'uploading' | 'uploaded' | 'processing' | 'indexed' | 'error' | 'cancelled';

export interface Document {
  id: string;
  filename: string;
  filesize: number;
  format: DocumentFormat;
  pageCount: number;
  wordCount: number;
  uploadedAt: Date;
  status: DocumentStatus;
  progress: number;
  errorMessage?: string;
}

export type MessageType = 'user' | 'bot';

export interface Source {
  legal_reference: string;
  section?: string;
  page?: number;
  quote?: string;
  relevance?: number;
}

export type SeverityLevel = 'critical' | 'high' | 'medium' | 'low' | 'info';

export interface LawRequirement {
  law_reference: string;
  section?: string;
  requirement_text?: string;
}

export interface ComplianceIssue {
  issue_id: string;
  severity: SeverityLevel;
  issue_description: string;
  contract_reference: string;
  law_requirements: LawRequirement[];
  recommendations: string[];
}

export interface ChatMessage {
  id: string;
  type: MessageType;
  content: string;
  timestamp: Date;
  sources?: Source[];
  complianceIssues?: ComplianceIssue[];
  isStreaming?: boolean;
}

export interface ComplianceReport {
  task_id: string;
  contract_id: string;
  law_ids: string[];
  issues: ComplianceIssue[];
  summary: string;
  timestamp: Date;
}

export type Language = 'cs' | 'en';

export type Theme = 'light' | 'dark';
