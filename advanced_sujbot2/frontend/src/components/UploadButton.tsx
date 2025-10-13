import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '@/utils/cn';
import { validateFile } from '@/utils/file';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';

interface UploadButtonProps {
  type: 'contract' | 'law';
  onUpload: (files: File[]) => Promise<void>;
  accept?: string;
  multiple?: boolean;
}

export const UploadButton: React.FC<UploadButtonProps> = ({
  type,
  onUpload,
  accept = '.pdf,.docx,.txt,.md,.odt,.rtf',
  multiple = true
}) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);

    const files = Array.from(e.dataTransfer.files);

    const invalidFile = files.find(file => !validateFile(file).valid);
    if (invalidFile) {
      setError(validateFile(invalidFile).error || 'Invalid file');
      return;
    }

    await onUpload(files);
    setIsOpen(false);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const files = Array.from(e.target.files || []);

    const invalidFile = files.find(file => !validateFile(file).valid);
    if (invalidFile) {
      setError(validateFile(invalidFile).error || 'Invalid file');
      return;
    }

    await onUpload(files);
    setIsOpen(false);
  };

  return (
    <>
      <Button
        onClick={() => setIsOpen(true)}
        className="w-full bg-black hover:bg-gray-800 text-white dark:bg-white dark:text-black dark:hover:bg-gray-200"
      >
        <Upload className="h-4 w-4 mr-2" />
        {t(`upload.${type}`)}
      </Button>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t(`upload.${type}`)}</DialogTitle>
            <DialogDescription>
              {t('upload.dragDropDescription')}
            </DialogDescription>
          </DialogHeader>

          <div
            className={cn(
              'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
              isDragging ? 'border-primary bg-primary/10' : 'border-gray-300 hover:border-primary dark:border-gray-600'
            )}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={handleClick}
          >
            <Upload className="mx-auto h-16 w-16 text-gray-400" />
            <p className="mt-4 text-sm font-medium">
              {t('upload.dragDrop')}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              {t('upload.orClickToSelect')}
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
              PDF, DOCX, TXT, MD, ODT, RTF
            </p>

            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept={accept}
              multiple={multiple}
              onChange={handleFileChange}
            />
          </div>

          {error && (
            <div className="mt-2 text-sm text-red-600 dark:text-red-400">
              {error}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};
