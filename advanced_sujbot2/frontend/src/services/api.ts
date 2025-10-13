import apiClient from './axios';
import { DocumentStatus, ComplianceReport } from '@/types';

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
): Promise<{
  document_id: string;
  filename: string;
  status: DocumentStatus;
  progress: number;
  metadata?: {
    page_count: number;
    word_count: number;
    chunk_count: number;
    format: string;
    indexed_at?: string;
  };
  error_message?: string;
}> => {
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
