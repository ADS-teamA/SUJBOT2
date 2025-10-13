import { DocumentFormat } from '@/types';

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export function getFileFormat(filename: string): DocumentFormat {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const formatMap: Record<string, DocumentFormat> = {
    'pdf': 'pdf',
    'docx': 'docx',
    'txt': 'txt',
    'md': 'md',
    'odt': 'odt',
    'rtf': 'rtf',
    'html': 'html',
    'htm': 'html',
    'epub': 'epub'
  };
  return formatMap[ext] || 'pdf';
}

const SUPPORTED_FORMATS: Record<string, string[]> = {
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

export function validateFile(file: File): { valid: boolean; error?: string } {
  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      error: `File size exceeds maximum (${MAX_FILE_SIZE / 1024 / 1024} MB)`
    };
  }

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

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
