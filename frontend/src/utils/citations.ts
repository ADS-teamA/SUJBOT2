/**
 * Citation Parsing Utilities
 *
 * Handles parsing and preprocessing of \cite{chunk_id} syntax in agent responses.
 */

/**
 * Citation marker regex pattern.
 * Matches: \cite{chunk_id} where chunk_id is alphanumeric with underscores/hyphens
 */
export const CITATION_REGEX = /\\cite\{([a-zA-Z0-9_-]+)\}/g;

/**
 * Extract all unique chunk_ids from content.
 *
 * @param content - Message content with \cite{} markers
 * @returns Array of unique chunk IDs
 *
 * @example
 * extractCitationIds("According to \\cite{BZ_VR1_L3_c5} and \\cite{BZ_VR1_L3_c6}...")
 * // Returns: ["BZ_VR1_L3_c5", "BZ_VR1_L3_c6"]
 */
export function extractCitationIds(content: string): string[] {
  const ids: string[] = [];
  let match;

  // Reset regex lastIndex (stateful due to /g flag)
  const regex = new RegExp(CITATION_REGEX.source, 'g');

  while ((match = regex.exec(content)) !== null) {
    ids.push(match[1]);
  }

  // Return unique IDs
  return [...new Set(ids)];
}

/**
 * Check if content contains any citation markers.
 *
 * @param content - Message content to check
 * @returns true if content has \cite{} markers
 */
export function hasCitations(content: string): boolean {
  return CITATION_REGEX.test(content);
}

/**
 * Preprocess content by replacing \cite{chunk_id} with HTML <cite> tags.
 *
 * This allows ReactMarkdown to pass through the cite elements which we
 * render with custom CitationLink components.
 *
 * @param content - Raw message content with \cite{} markers
 * @returns Preprocessed content with <cite> HTML tags
 *
 * @example
 * preprocessCitations("Safety margin is 5mm \\cite{BZ_VR1_L3_c5}.")
 * // Returns: 'Safety margin is 5mm <cite data-chunk-id="BZ_VR1_L3_c5">[BZ_VR1_L3_c5]</cite>.'
 */
export function preprocessCitations(content: string): string {
  return content.replace(
    CITATION_REGEX,
    '<cite data-chunk-id="$1">[$1]</cite>'
  );
}

/**
 * Format citation display text in short format.
 *
 * @param documentId - Document identifier
 * @param pageNumber - Page number (can be null)
 * @returns Formatted string like "[BZ_VR1:12]" or "[BZ_VR1]"
 */
export function formatCitationShort(
  documentId: string,
  pageNumber: number | null
): string {
  if (pageNumber !== null && pageNumber > 0) {
    return `[${documentId}:${pageNumber}]`;
  }
  return `[${documentId}]`;
}

/**
 * Format citation tooltip text.
 *
 * @param sectionPath - Section path/breadcrumb
 * @param sectionTitle - Section title
 * @returns Formatted tooltip string
 */
export function formatCitationTooltip(
  sectionPath: string | null,
  sectionTitle: string | null
): string {
  if (sectionPath) {
    return sectionPath;
  }
  if (sectionTitle) {
    return sectionTitle;
  }
  return 'Click to view in PDF';
}
