import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Document, DocumentStatus } from '@/types';
import { uploadDocument as uploadDocumentAPI, deleteDocument, getDocumentStatus } from '@/services/api';
import { generateId } from '@/utils/file';
import { useToastStore } from './toastStore';

interface DocumentStore {
  contracts: Document[];
  laws: Document[];
  selectedDocumentIds: string[]; // NEW: Track selected documents for queries
  progressInterpolators: Map<string, NodeJS.Timeout>; // Track interpolation timers

  uploadDocument: (file: File, type: 'contract' | 'law') => Promise<void>;
  removeDocument: (id: string, type: 'contract' | 'law') => void;
  updateDocumentStatus: (id: string, status: DocumentStatus, progress?: number) => void;
  updateDocumentMetadata: (id: string, pageCount: number, wordCount: number) => void;
  setErrorMessage: (id: string, errorMessage: string) => void;
  startProgressInterpolation: (id: string, currentProgress: number, targetProgress: number) => void;
  stopProgressInterpolation: (id: string) => void;

  // NEW: Document selection methods
  toggleDocumentSelection: (id: string) => void;
  selectAllDocuments: () => void;
  deselectAllDocuments: () => void;
  isDocumentSelected: (id: string) => boolean;
  getSelectedDocuments: () => Document[];

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
        selectedDocumentIds: [], // Initialize empty selection
        progressInterpolators: new Map(), // Initialize interpolation map

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

            // Show upload started toast
            useToastStore.getState().info('toast.upload.started', { filename: file.name });

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
                setTimeout(pollStatus, 500);  // Poll every 500ms (reduced from 200ms to prevent race conditions)
              } else if (statusData.status === 'indexed') {
                // Show success toast when indexing complete
                useToastStore.getState().success('toast.upload.success', { filename: file.name });
              } else if (statusData.status === 'cancelled') {
                // Document processing was cancelled - stop polling
                useToastStore.getState().info('toast.upload.cancelled', { filename: file.name });
              } else if (statusData.status === 'error' && statusData.error_message) {
                get().setErrorMessage(backendDocId, statusData.error_message);
                useToastStore.getState().error('toast.upload.error', { error: statusData.error_message });
              }
            };

            await pollStatus();

          } catch (error: any) {
            get().updateDocumentStatus(tempId, 'error');
            get().setErrorMessage(tempId, error.message);
            useToastStore.getState().error('toast.upload.error', { error: error.message });
          }
        },

        removeDocument: (id, type) => {
          const doc = get().getDocumentById(id);

          // IMPORTANT: Stop any progress interpolation immediately
          get().stopProgressInterpolation(id);

          // Remove document from UI immediately (optimistic update)
          set((state) => ({
            [type === 'contract' ? 'contracts' : 'laws']:
              state[type === 'contract' ? 'contracts' : 'laws'].filter(d => d.id !== id),
            selectedDocumentIds: state.selectedDocumentIds.filter(docId => docId !== id)
          }));

          // Call backend to delete (with task cancellation, index cleanup, etc.)
          deleteDocument(id)
            .then(() => {
              useToastStore.getState().success('toast.document.removed', { filename: doc?.filename || '' });
            })
            .catch((error) => {
              // If document doesn't exist (404), that's fine - it's already gone
              if (error?.response?.status === 404) {
                console.log(`Document ${id} already deleted from backend`);
                useToastStore.getState().success('toast.document.removed', { filename: doc?.filename || '' });
                return;
              }

              // For other errors, rollback the optimistic update
              if (doc) {
                set((state) => ({
                  [type === 'contract' ? 'contracts' : 'laws']: [
                    ...state[type === 'contract' ? 'contracts' : 'laws'],
                    doc
                  ]
                }));
              }
              useToastStore.getState().error('toast.document.removeError', { error: error.message });
            });
        },

        updateDocumentStatus: (id, status, progress) => {
          // Stop any existing interpolation for this document
          get().stopProgressInterpolation(id);

          set((state) => {
            const updateDocs = (docs: Document[]) =>
              docs.map(d => {
                if (d.id === id) {
                  const currentProgress = d.progress || 0;
                  const newProgress = progress ?? d.progress;

                  // IMPORTANT: Ignore progress updates that go backwards (race condition protection)
                  // Only update if new progress is greater or status changed to final state
                  const isFinalState = status === 'indexed' || status === 'error';
                  const shouldUpdate = newProgress >= currentProgress || isFinalState;

                  if (!shouldUpdate && !isFinalState) {
                    // Ignore backwards progress update
                    return d;
                  }

                  // Start interpolation ONLY if there's a significant gap and we're not at final state
                  if (!isFinalState && newProgress !== undefined && newProgress > currentProgress + 5) {
                    // Will interpolate in next tick
                    setTimeout(() => get().startProgressInterpolation(id, currentProgress, newProgress), 0);
                  }

                  return { ...d, status, progress: isFinalState ? 100 : newProgress };
                }
                return d;
              });

            return {
              contracts: updateDocs(state.contracts),
              laws: updateDocs(state.laws)
            };
          });
        },

        startProgressInterpolation: (id, currentProgress, targetProgress) => {
          const state = get();

          // Clear any existing interpolation
          state.stopProgressInterpolation(id);

          let progress = currentProgress;
          const totalSteps = 30; // More steps for smoother animation
          const step = (targetProgress - currentProgress) / totalSteps;
          let currentStep = 0;

          const intervalId = setInterval(() => {
            currentStep++;

            // Use easing function for smooth acceleration/deceleration
            const easeFactor = currentStep / totalSteps;
            const easeProgress = easeFactor < 0.5
              ? 2 * easeFactor * easeFactor  // Ease in
              : 1 - Math.pow(-2 * easeFactor + 2, 2) / 2;  // Ease out

            progress = currentProgress + (targetProgress - currentProgress) * easeProgress;

            if (currentStep >= totalSteps || progress >= targetProgress) {
              progress = targetProgress;
              get().stopProgressInterpolation(id);
            }

            set((state) => {
              const updateDocs = (docs: Document[]) =>
                docs.map(d => d.id === id ? { ...d, progress: Math.round(progress) } : d);

              return {
                contracts: updateDocs(state.contracts),
                laws: updateDocs(state.laws)
              };
            });
          }, 33); // ~30 FPS for ultra-smooth animation

          set((state) => ({
            progressInterpolators: new Map(state.progressInterpolators).set(id, intervalId)
          }));
        },

        stopProgressInterpolation: (id) => {
          const state = get();
          const intervalId = state.progressInterpolators.get(id);

          if (intervalId) {
            clearInterval(intervalId);
            const newMap = new Map(state.progressInterpolators);
            newMap.delete(id);
            set({ progressInterpolators: newMap });
          }
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

        // NEW: Document selection methods
        toggleDocumentSelection: (id) => {
          set((state) => {
            const isSelected = state.selectedDocumentIds.includes(id);
            return {
              selectedDocumentIds: isSelected
                ? state.selectedDocumentIds.filter(docId => docId !== id)
                : [...state.selectedDocumentIds, id]
            };
          });
        },

        selectAllDocuments: () => {
          const state = get();
          const allIndexedDocs = [...state.contracts, ...state.laws]
            .filter(d => d.status === 'indexed')
            .map(d => d.id);
          set({ selectedDocumentIds: allIndexedDocs });
        },

        deselectAllDocuments: () => {
          set({ selectedDocumentIds: [] });
        },

        isDocumentSelected: (id) => {
          return get().selectedDocumentIds.includes(id);
        },

        getSelectedDocuments: () => {
          const state = get();
          const allDocs = [...state.contracts, ...state.laws];
          return allDocs.filter(d => state.selectedDocumentIds.includes(d.id));
        },

        // Resume polling for documents on page load
        resumePolling: () => {
          const state = get();
          const allDocs = [...state.contracts, ...state.laws];

          allDocs.forEach(doc => {
            // Check if document exists on backend (for all documents, not just processing ones)
            const checkDocument = async () => {
              try {
                await getDocumentStatus(doc.id);
              } catch (error: any) {
                // If document doesn't exist (404), remove it from localStorage
                if (error?.response?.status === 404) {
                  console.log(`Document ${doc.id} no longer exists on backend, removing from localStorage`);
                  const state = get();
                  const isContract = state.contracts.some(d => d.id === doc.id);
                  const type = isContract ? 'contract' : 'law';

                  // Remove from store silently (no API call, no toast)
                  set((state) => ({
                    [type === 'contract' ? 'contracts' : 'laws']:
                      state[type === 'contract' ? 'contracts' : 'laws'].filter(d => d.id !== doc.id),
                    selectedDocumentIds: state.selectedDocumentIds.filter(docId => docId !== doc.id)
                  }));
                }
              }
            };

            checkDocument();

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
                    setTimeout(pollStatus, 500);  // Poll every 500ms (reduced from 200ms to prevent race conditions)
                  } else if (statusData.status === 'cancelled') {
                    // Document processing was cancelled - stop polling
                    // Document will be removed from UI soon
                  } else if (statusData.status === 'error' && statusData.error_message) {
                    get().setErrorMessage(doc.id, statusData.error_message);
                  }
                } catch (error: any) {
                  // If document doesn't exist (404), remove it from store
                  if (error?.response?.status === 404) {
                    console.log(`Document ${doc.id} no longer exists on backend, removing from localStorage`);
                    const state = get();
                    const isContract = state.contracts.some(d => d.id === doc.id);
                    const type = isContract ? 'contract' : 'law';

                    // Remove from store without calling deleteDocument API (document already gone)
                    set((state) => ({
                      [type === 'contract' ? 'contracts' : 'laws']:
                        state[type === 'contract' ? 'contracts' : 'laws'].filter(d => d.id !== doc.id)
                    }));
                  } else {
                    console.error(`Failed to poll status for ${doc.id}:`, error);
                  }
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
          laws: state.laws,
          selectedDocumentIds: state.selectedDocumentIds
        })
      }
    )
  )
);
