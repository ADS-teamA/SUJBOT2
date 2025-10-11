import React from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, X, Eye } from 'lucide-react';
import { Document, DocumentStatus } from '@/types';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
import { formatFileSize } from '@/utils/file';
import { useDocumentStore } from '@/stores/documentStore';

interface DocumentCardProps {
  document: Document;
  onRemove: (id: string) => void;
  onPreview?: (id: string) => void;
}

const getStatusVariant = (status: DocumentStatus): 'default' | 'secondary' | 'destructive' => {
  switch (status) {
    case 'indexed':
      return 'default';
    case 'uploaded':
    case 'processing':
    case 'uploading':
      return 'secondary';
    case 'error':
      return 'destructive';
    default:
      return 'secondary';
  }
};

export const DocumentCard: React.FC<DocumentCardProps> = ({ document, onRemove, onPreview }) => {
  const { t } = useTranslation();
  const { isDocumentSelected, toggleDocumentSelection } = useDocumentStore();

  const isIndexed = document.status === 'indexed';
  const isSelected = isDocumentSelected(document.id);

  return (
    <Card className={`p-2 hover:shadow-md transition-all ${isSelected ? 'ring-2 ring-primary' : ''}`}>
      <div className="flex items-start justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          {isIndexed && (
            <Checkbox
              checked={isSelected}
              onCheckedChange={() => toggleDocumentSelection(document.id)}
              className="cursor-pointer"
            />
          )}
          <FileText className="h-4 w-4 text-gray-500 flex-shrink-0" />
        </div>
        <div className="flex items-center gap-1">
          <Badge variant={getStatusVariant(document.status)} className="text-xs px-1.5 py-0.5">
            {t(`document.status.${document.status}`)}
          </Badge>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => onRemove(document.id)}
            className="h-5 w-5 flex-shrink-0"
            title={t('document.remove')}
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <h3 className="text-sm font-medium truncate mb-1.5" title={document.filename}>
        {document.filename}
      </h3>

      <div className="text-xs text-gray-600 dark:text-gray-400 mb-1.5">
        <div className="flex justify-between gap-2">
          <span>{document.pageCount} {t('document.pages')}</span>
          <span>{formatFileSize(document.filesize)}</span>
        </div>
        {document.wordCount > 0 && (
          <div className="mt-0.5">
            {document.wordCount.toLocaleString()} {t('document.words')}
          </div>
        )}
      </div>

      {(document.status === 'processing' || document.status === 'uploading' || document.status === 'uploaded') && (
        <Progress value={document.progress} className="mb-1.5" />
      )}

      {document.status === 'error' && document.errorMessage && (
        <div className="mb-1.5 text-xs text-red-600 dark:text-red-400">
          {document.errorMessage}
        </div>
      )}

      {onPreview && (
        <Button
          size="sm"
          variant="outline"
          onClick={() => onPreview(document.id)}
          className="w-full h-7 text-xs"
        >
          <Eye className="h-3 w-3 mr-1" />
          {t('document.preview')}
        </Button>
      )}
    </Card>
  );
};
