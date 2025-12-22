/**
 * Hook: usePopoverPosition
 *
 * Calculates optimal popover position relative to anchor element.
 * Automatically flips above/below based on viewport constraints.
 *
 * Usage:
 * const position = usePopoverPosition(anchorRect, 200, 320, 8);
 * if (position) {
 *   style={{ top: position.top, left: position.left }}
 * }
 */

import { useMemo } from 'react';

export interface PopoverPosition {
  top: number;
  left: number;
  placement: 'top' | 'bottom';
  arrowLeft: number; // Arrow horizontal position relative to popover
}

interface UsePopoverPositionOptions {
  anchorRect: DOMRect | null;
  popoverHeight: number;
  popoverWidth: number;
  gap?: number; // Space between anchor and popover (default: 8px)
  viewportPadding?: number; // Minimum distance from viewport edges (default: 8px)
}

export function usePopoverPosition({
  anchorRect,
  popoverHeight,
  popoverWidth,
  gap = 8,
  viewportPadding = 8,
}: UsePopoverPositionOptions): PopoverPosition | null {
  return useMemo(() => {
    if (!anchorRect) return null;

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Calculate available space above and below anchor
    const spaceBelow = viewportHeight - anchorRect.bottom - gap;
    const spaceAbove = anchorRect.top - gap;

    // Determine vertical placement: prefer below, flip to above if needed
    const placement: 'top' | 'bottom' =
      spaceBelow >= popoverHeight || spaceBelow >= spaceAbove ? 'bottom' : 'top';

    // Calculate vertical position
    const top =
      placement === 'bottom'
        ? anchorRect.bottom + gap
        : anchorRect.top - popoverHeight - gap;

    // Calculate horizontal position: center on anchor, clamp to viewport
    const anchorCenter = anchorRect.left + anchorRect.width / 2;
    let left = anchorCenter - popoverWidth / 2;

    // Clamp to viewport edges
    const minLeft = viewportPadding;
    const maxLeft = viewportWidth - popoverWidth - viewportPadding;
    left = Math.max(minLeft, Math.min(maxLeft, left));

    // Calculate arrow position relative to popover
    // Arrow should point to anchor center
    const arrowLeft = Math.max(
      12, // Min distance from left edge
      Math.min(
        popoverWidth - 12, // Max distance from left edge
        anchorCenter - left
      )
    );

    return { top, left, placement, arrowLeft };
  }, [anchorRect, popoverHeight, popoverWidth, gap, viewportPadding]);
}
