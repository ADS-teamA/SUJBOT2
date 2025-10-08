import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Document, DocumentStatus } from '@/types';
import { uploadDocument as uploadDocumentAPI, deleteDocument, getDocumentStatus } from '@/services/api';
import { generateId } from '@/utils/file';

interface DocumentStore {
  contracts: Document[];
  laws: Document[];

  uploadDocument: (file: File, type: 'contract' | 'law') => Promise<void>;
  removeDocument: (id: string, type: 'contract' | 'law') => void;
  updateDocumentStatus: (id: string, status: DocumentStatus, progress?: number) => void;
  updateDocumentMetadata: (id: string, pageCount: number, wordCount: number) => void;
  setErrorMessage: (id: string, errorMessage: string) => void;

  getDocumentById: (id: string) => Document | undefined;
  getDocumentsByType: (type: 'contract' | 'law') => Document[];
  resumePolling: () => void;
}

export const useDocumentStore = create<DocumentStore>()(
  devtools(
    persist(
      (set, get) => ({
        contracts: [],
        laws: [],

        uploadDocument: async (file, type) => {
          // Generate temporary ID for initial display
          const tempId = generateId();
          const newDoc: Document = {
            id: tempId,
            filename: file.name,
            filesize: file.size,
            format: file.name.split('.').pop()?.toLowerCase() as any || 'pdf',
            pageCount: 0,
            wordCount: 0,
            uploadedAt: new Date(),
            status: 'uploading',
            progress: 0
          };

          set((state) => ({
            [type === 'contract' ? 'contracts' : 'laws']: [
              ...state[type === 'contract' ? 'contracts' : 'laws'],
              newDoc
            ]
          }));

          try {
            // Upload and get backend's document ID
            const result = await uploadDocumentAPI(file, type, (progress) => {
              get().updateDocumentStatus(tempId, 'uploading', progress);
            });

            const backendDocId = result.document_id;

            // Replace temporary ID with backend's ID
            set((state) => {
              const docList = type === 'contract' ? 'contracts' : 'laws';
              return {
                [docList]: state[docList].map(d =>
                  d.id === tempId ? { ...d, id: backendDocId, status: 'processing', progress: 0 } : d
                )
              };
            });

            // Poll status with correct backend ID
            const pollStatus = async () => {
              const statusData = await getDocumentStatus(backendDocId);

              // Update status and progress
              get().updateDocumentStatus(backendDocId, statusData.status, statusData.progress);

              // Update metadata if available
              if (statusData.metadata) {
                get().updateDocumentMetadata(
                  backendDocId,
                  statusData.metadata.page_count,
                  statusData.metadata.word_count
                );
              }

              // Continue polling if still processing
              if (statusData.status === 'processing' || statusData.status === 'uploading' || statusData.status === 'uploaded') {
                setTimeout(pollStatus, 500);  // Poll every 500ms for faster updates
              } else if (statusData.status === 'error' && statusData.error_message) {
                get().setErrorMessage(backendDocId, statusData.error_message);
              }
            };

            await pollStatus();

          } catch (error: any) {
            get().updateDocumentStatus(tempId, 'error');
            get().setErrorMessage(tempId, error.message);
          }
        },

        removeDocument: (id, type) => {
          set((state) => ({
            [type === 'contract' ? 'contracts' : 'laws']:
              state[type === 'contract' ? 'contracts' : 'laws'].filter(d => d.id !== id)
          }));

          deleteDocument(id).catch(console.error);
        },

        updateDocumentStatus: (id, status, progress) => {
          set((state) => {
            const updateDocs = (docs: Document[]) =>
              docs.map(d => d.id === id ? { ...d, status, progress: progress ?? d.progress } : d);

            return {
              contracts: updateDocs(state.contracts),
              laws: updateDocs(state.laws)
            };
          });
        },

        updateDocumentMetadata: (id, pageCount, wordCount) => {
          set((state) => {
            const updateDocs = (docs: Document[]) =>
              docs.map(d => d.id === id ? { ...d, pageCount, wordCount } : d);

            return {
              contracts: updateDocs(state.contracts),
              laws: updateDocs(state.laws)
            };
          });
        },

        setErrorMessage: (id, errorMessage) => {
          set((state) => {
            const updateDocs = (docs: Document[]) =>
              docs.map(d => d.id === id ? { ...d, errorMessage } : d);

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
        },

        // Resume polling for documents on page load
        resumePolling: () => {
          const state = get();
          const allDocs = [...state.contracts, ...state.laws];

          allDocs.forEach(doc => {
            if (doc.status === 'uploaded' || doc.status === 'processing' || doc.status === 'uploading') {
              // Resume polling for this document
              const pollStatus = async () => {
                try {
                  const statusData = await getDocumentStatus(doc.id);
                  get().updateDocumentStatus(doc.id, statusData.status, statusData.progress);

                  if (statusData.metadata) {
                    get().updateDocumentMetadata(
                      doc.id,
                      statusData.metadata.page_count,
                      statusData.metadata.word_count
                    );
                  }

                  if (statusData.status === 'processing' || statusData.status === 'uploading' || statusData.status === 'uploaded') {
                    setTimeout(pollStatus, 1000);
                  } else if (statusData.status === 'error' && statusData.error_message) {
                    get().setErrorMessage(doc.id, statusData.error_message);
                  }
                } catch (error) {
                  console.error(`Failed to poll status for ${doc.id}:`, error);
                }
              };

              pollStatus();
            }
          });
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
