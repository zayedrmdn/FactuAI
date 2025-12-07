'use client';

import { buildStyles, CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

interface OverallScoreProps {
  /** 0–100 confidence score of the prediction */
  readonly score: number;
  readonly title?: string;
  readonly className?: string;
}

export default function OverallScore({
  score,
  title = 'Overall Confidence',
  className = '',
}: Readonly<OverallScoreProps>) {
  // Handle NaN/undefined values
  const normalizedScore =
    Number.isNaN(score) || score === undefined ? 0 : Math.max(0, Math.min(100, score));

  // Color based on prediction confidence - using CSS variables
  const getColorAndLabel = (score: number) => {
    if (score >= 80) return { color: 'oklch(var(--score-very-high))', label: 'Very High' };
    if (score >= 60) return { color: 'oklch(var(--score-high))', label: 'High' };
    if (score >= 40) return { color: 'oklch(var(--score-medium))', label: 'Medium' };
    if (score >= 20) return { color: 'oklch(var(--score-low))', label: 'Low' };
    return { color: 'oklch(var(--score-very-low))', label: 'Very Low' };
  };

  const { color, label } = getColorAndLabel(normalizedScore);
  const trailColor = 'oklch(var(--score-trail))';

  // Check if this is a small inline version (for QA cards)
  const isSmall = className.includes('w-12') || className.includes('h-12');

  if (isSmall) {
    return (
      <div className={`${className}`} title={`${title}: ${label} (${normalizedScore.toFixed(0)}%)`}>
        <CircularProgressbar
          value={normalizedScore}
          text={`${normalizedScore.toFixed(0)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor,
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
          value={score}
          text={`${score.toFixed(0)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor,
            pathTransitionDuration: 0.5,
            textSize: '16px',
          })}
        />
      </div>

      <div className="w-full text-center space-y-2">
        <div className="text-xs font-medium text-gray-600 dark:text-gray-400">{title}</div>
        <div className="text-sm font-semibold" style={{ color }}>
          {label} ({score.toFixed(0)}%)
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
