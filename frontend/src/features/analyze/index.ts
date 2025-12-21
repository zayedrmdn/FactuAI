/**
 * Analyze Feature Module
 *
 * Centralized export for all fact-checking UI components.
 * This is the single entry point for the analyze domain.
 *
 * Usage:
 * ```tsx
 * import {
 *   InputCard,
 *   ResultsView,
 *   ClaimCard,
 *   QAResultCard,
 *   InputTabs,
 *   TextInput,
 *   ImageInput,
 *   VideoInput,
 * } from '@/features/analyze';
 * ```
 */

// Input Components
export { default as InputCard } from './components/InputCard';
export { default as InputTabs } from './components/InputTabs';
export { default as TextInput } from './components/TextInput';
export { default as ImageInput } from './components/ImageInput';
export { default as VideoInput } from './components/VideoInput';

// Result Components
export { default as ResultsView } from './components/ResultsView';
export { default as ClaimCard } from './components/ClaimCard';
export { QAResultCard } from './components/QAResultCard';

// Analysis Mode
export { AnalysisModeToggle } from './components/AnalysisModeToggle';
