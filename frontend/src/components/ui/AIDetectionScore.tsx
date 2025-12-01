"use client";

import { buildStyles, CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface AIDetectionScoreProps {
  /** 0–100 AI probability score */
  score: number;
  /** Optional error message if detection failed */
  error?: string;
  className?: string;
}

export default function AIDetectionScore({ score, error, className = "" }: AIDetectionScoreProps) {
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

  // Color based on AI probability with better thresholds
  const getColorAndLabel = (score: number) => {
    if (score >= 80) return { color: "#dc2626", label: "Very Likely AI", risk: "high" };
    if (score >= 60) return { color: "#ea580c", label: "Likely AI", risk: "medium-high" };
    if (score >= 40) return { color: "#d97706", label: "Possibly AI", risk: "medium" };
    if (score >= 20) return { color: "#65a30d", label: "Probably Human", risk: "low" };
    return { color: "#16a34a", label: "Very Likely Human", risk: "very-low" };
  };

  const { color, label } = getColorAndLabel(score);

  return (
    <div className={`max-w-xs mx-auto mb-6 ${className}`}>
      <div className="w-24 h-24 mx-auto mb-3">
        <CircularProgressbar
          value={score}
          text={`${score.toFixed(1)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor: "#e5e7eb",
            pathTransitionDuration: 0.5,
            textSize: "16px",
          })}
        />
      </div>

      <div className="text-center space-y-2">
        <div className="text-xs font-medium text-gray-600 dark:text-gray-400">
          AI Detection Score
        </div>
        <div className="text-sm font-semibold" style={{ color }}>
          {label}
        </div>
        
        {/* Legend */}
        <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1 pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-red-600 rounded-full" />
            <span>Very Likely AI (≥80%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-orange-600 rounded-full" />
            <span>Likely AI (60-79%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-amber-600 rounded-full" />
            <span>Possibly AI (40-59%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-lime-600 rounded-full" />
            <span>Probably Human (20-39%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-green-600 rounded-full" />
            <span>Very Likely Human (&lt;20%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
