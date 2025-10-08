import React from 'react';
import { useTranslation } from 'react-i18next';
import { Document } from '@/types';
import { DocumentCard } from './DocumentCard';
import { UploadButton } from './UploadButton';
import { useDocumentStore } from '@/stores/documentStore';

interface DocumentPanelProps {
  type: 'contract' | 'law';
}

export const DocumentPanel: React.FC<DocumentPanelProps> = ({ type }) => {
  const { t } = useTranslation();
  const { contracts, laws, uploadDocument, removeDocument } = useDocumentStore();

  const documents = type === 'contract' ? contracts : laws;

  const handleUpload = async (files: File[]) => {
    for (const file of files) {
      await uploadDocument(file, type);
    }
  };

  const handleRemove = (id: string) => {
    removeDocument(id, type);
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-white via-gray-50 to-white dark:from-black dark:via-gray-950 dark:to-black border-r border-gray-200 dark:border-gray-800 transition-all duration-500">
      <div className="p-4 border-b border-gray-200 dark:border-gray-800 backdrop-blur-sm">
        <h2 className="text-lg font-semibold mb-4 bg-gradient-to-r from-black to-gray-600 dark:from-white dark:to-gray-400 bg-clip-text text-transparent">
          {t(`panel.${type === 'contract' ? 'contracts' : 'laws'}`)}
        </h2>
        <UploadButton type={type} onUpload={handleUpload} />
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {documents.length === 0 ? (
          <div className="text-center text-gray-500 dark:text-gray-400 mt-8 animate-fade-in">
            {t('panel.noDocuments')}
          </div>
        ) : (
          documents.map((doc, index) => (
            <div key={doc.id} className="animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
              <DocumentCard
                document={doc}
                onRemove={handleRemove}
              />
            </div>
          ))
        )}
      </div>
    </div>
  );
};
