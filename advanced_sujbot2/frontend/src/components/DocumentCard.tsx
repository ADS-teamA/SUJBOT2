import React from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, X, Eye } from 'lucide-react';
import { Document, DocumentStatus } from '@/types';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { formatFileSize } from '@/utils/file';

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

  return (
    <Card className="p-4 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between mb-2">
        <FileText className="h-5 w-5 text-gray-500" />
        <Badge variant={getStatusVariant(document.status)}>
          {t(`document.status.${document.status}`)}
        </Badge>
      </div>

      <h3 className="font-medium truncate" title={document.filename}>
        {document.filename}
      </h3>

      <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
        <div className="flex justify-between">
          <span>{document.pageCount} {t('document.pages')}</span>
          <span>{formatFileSize(document.filesize)}</span>
        </div>
        {document.wordCount > 0 && (
          <div className="mt-1">
            {document.wordCount.toLocaleString()} {t('document.words')}
          </div>
        )}
      </div>

      {(document.status === 'processing' || document.status === 'uploading' || document.status === 'uploaded') && (
        <Progress value={document.progress} className="mt-3" />
      )}

      {document.status === 'error' && document.errorMessage && (
        <div className="mt-3 text-sm text-red-600 dark:text-red-400">
          {document.errorMessage}
        </div>
      )}

      <div className="flex gap-2 mt-3">
        {onPreview && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => onPreview(document.id)}
            className="flex-1"
          >
            <Eye className="h-4 w-4 mr-1" />
            {t('document.preview')}
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onRemove(document.id)}
        >
          <X className="h-4 w-4 mr-1" />
          {t('document.remove')}
        </Button>
      </div>
    </Card>
  );
};
