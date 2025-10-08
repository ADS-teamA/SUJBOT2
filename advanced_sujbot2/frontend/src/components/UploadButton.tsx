import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '@/utils/cn';
import { validateFile } from '@/utils/file';

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
  };

  return (
    <div>
      <div
        className={cn(
          'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
          isDragging ? 'border-primary bg-primary/10' : 'border-gray-300 hover:border-primary dark:border-gray-600'
        )}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <Upload className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm font-medium">
          {t(`upload.${type}`)}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {t('upload.dragDrop')}
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
    </div>
  );
};
