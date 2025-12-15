/**
 * History Feature Module
 *
 * Centralized export for history UI components and state management.
 * This is the single entry point for the history domain.
 *
 * Usage:
 * ```tsx
 * import { HistoryPanel, HistoryItem, useHistory } from '@/features/history';
 * ```
 */

// Components
export { default as HistoryPanel } from './components/HistoryPanel';
export { default as HistoryItem } from './components/HistoryItem';

// Hooks
export { useHistory } from './hooks/useHistory';
