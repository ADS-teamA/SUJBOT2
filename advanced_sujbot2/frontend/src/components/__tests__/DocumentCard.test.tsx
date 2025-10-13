import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentCard } from '../DocumentCard';
import { Document } from '@/types';

const mockDocument: Document = {
  id: '1',
  filename: 'test.pdf',
  filesize: 1024000,
  format: 'pdf',
  pageCount: 100,
  wordCount: 5000,
  uploadedAt: new Date(),
  status: 'indexed',
  progress: 100
};

describe('DocumentCard', () => {
  it('renders document metadata correctly', () => {
    render(<DocumentCard document={mockDocument} onRemove={vi.fn()} />);

    expect(screen.getByText('test.pdf')).toBeInTheDocument();
    expect(screen.getByText(/100/)).toBeInTheDocument();
    expect(screen.getByText(/1\.0 MB/)).toBeInTheDocument();
  });

  it('shows progress bar when processing', () => {
    const processingDoc: Document = {
      ...mockDocument,
      status: 'processing',
      progress: 50
    };

    render(<DocumentCard document={processingDoc} onRemove={vi.fn()} />);

    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toBeInTheDocument();
  });

  it('calls onRemove when remove button is clicked', () => {
    const handleRemove = vi.fn();
    render(<DocumentCard document={mockDocument} onRemove={handleRemove} />);

    const removeButton = screen.getByRole('button', { name: /remove/i });
    fireEvent.click(removeButton);

    expect(handleRemove).toHaveBeenCalledWith('1');
  });

  it('shows error message when status is error', () => {
    const errorDoc: Document = {
      ...mockDocument,
      status: 'error',
      errorMessage: 'Upload failed'
    };

    render(<DocumentCard document={errorDoc} onRemove={vi.fn()} />);

    expect(screen.getByText('Upload failed')).toBeInTheDocument();
  });
});
