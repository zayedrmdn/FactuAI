'use client';

import { buildStyles, CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

// ========================================================================================
// SHARED UTILITIES
// ========================================================================================

interface ScoreColor {
  readonly color: string;
  readonly label: string;
}

/** Get color and label for AI detection scores (inverted scale: high score = AI) */
function getAIDetectionColorAndLabel(score: number): ScoreColor {
  if (score >= 80) return { color: 'oklch(var(--score-very-low))', label: 'Very Likely AI' };
  if (score >= 60) return { color: 'oklch(var(--score-low))', label: 'Likely AI' };
  if (score >= 40) return { color: 'oklch(var(--score-medium))', label: 'Possibly AI' };
  if (score >= 20) return { color: 'oklch(var(--score-high))', label: 'Probably Human' };
  return { color: 'oklch(var(--score-very-high))', label: 'Very Likely Human' };
}

/** Get color and label for confidence scores (normal scale: high score = confident) */
function getConfidenceColorAndLabel(score: number): ScoreColor {
  if (score >= 80) return { color: 'oklch(var(--score-very-high))', label: 'Very High' };
  if (score >= 60) return { color: 'oklch(var(--score-high))', label: 'High' };
  if (score >= 40) return { color: 'oklch(var(--score-medium))', label: 'Medium' };
  if (score >= 20) return { color: 'oklch(var(--score-low))', label: 'Low' };
  return { color: 'oklch(var(--score-very-low))', label: 'Very Low' };
}

const TRAIL_COLOR = 'oklch(var(--score-trail))';

// ========================================================================================
// AI DETECTION SCORE COMPONENT
// ========================================================================================

interface AIDetectionScoreProps {
  readonly score: number;
  readonly error?: string | undefined;
  readonly className?: string;
}

export function AIDetectionScore({ score, error, className = '' }: AIDetectionScoreProps) {
  if (error) {
    return (
      <div className={`max-w-xs mx-auto mb-6 ${className}`}>
        <div className="text-center text-sm text-muted-foreground p-4 border rounded-lg bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800">
          <p className="font-medium mb-1 text-gray-900 dark:text-gray-100">AI Detection</p>
          <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  const { color, label } = getAIDetectionColorAndLabel(score);

  return (
    <div className={`w-full flex flex-col items-center ${className}`}>
      <div className="w-24 h-24 mb-3">
        <CircularProgressbar
          value={score}
          text={`${score.toFixed(1)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor: TRAIL_COLOR,
            pathTransitionDuration: 0.5,
            textSize: '16px',
          })}
        />
      </div>

      <div className="w-full text-center space-y-2">
        <div className="text-xs font-medium text-gray-600 dark:text-gray-400">
          AI Detection Score
        </div>
        <div className="text-sm font-semibold" style={{ color }}>
          {label}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 w-full mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-very-low))' }}
            />
            <span>Very Likely AI (≥80%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-low))' }}
            />
            <span>Likely AI (60-79%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-medium))' }}
            />
            <span>Possibly AI (40-59%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-high))' }}
            />
            <span>Probably Human (20-39%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-very-high))' }}
            />
            <span>Very Likely Human (&lt;20%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ========================================================================================
// OVERALL CONFIDENCE SCORE COMPONENT
// ========================================================================================

interface OverallScoreProps {
  readonly score: number;
  readonly title?: string;
  readonly className?: string;
}

export function OverallScore({
  score,
  title = 'Overall Confidence',
  className = '',
}: OverallScoreProps) {
  const normalizedScore =
    Number.isNaN(score) || score === undefined ? 0 : Math.max(0, Math.min(100, score));
  const { color, label } = getConfidenceColorAndLabel(normalizedScore);

  const isSmall = className.includes('w-12') || className.includes('h-12');

  if (isSmall) {
    return (
      <div className={className} title={`${title}: ${label} (${normalizedScore.toFixed(0)}%)`}>
        <CircularProgressbar
          value={normalizedScore}
          text={`${normalizedScore.toFixed(0)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor: TRAIL_COLOR,
            pathTransitionDuration: 0.5,
            textSize: '24px',
          })}
        />
      </div>
    );
  }

  return (
    <div className={`w-full flex flex-col items-center ${className}`}>
      <div className="w-24 h-24 mb-3">
        <CircularProgressbar
          value={normalizedScore}
          text={`${normalizedScore.toFixed(0)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor: TRAIL_COLOR,
            pathTransitionDuration: 0.5,
            textSize: '16px',
          })}
        />
      </div>

      <div className="w-full text-center space-y-2">
        <div className="text-xs font-medium text-gray-600 dark:text-gray-400">{title}</div>
        <div className="text-sm font-semibold" style={{ color }}>
          {label} ({normalizedScore.toFixed(0)}%)
        </div>

        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 w-full mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-very-high))' }}
            />
            <span>Very High (≥80%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-high))' }}
            />
            <span>High (60-79%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-medium))' }}
            />
            <span>Medium (40-59%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-low))' }}
            />
            <span>Low (20-39%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: 'oklch(var(--score-very-low))' }}
            />
            <span>Very Low (&lt;20%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Default exports for backwards compatibility
export default OverallScore;
