"use client";

import { buildStyles, CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface OverallScoreProps {
  /** 0–100 confidence score of the prediction */
  score: number;
  title?: string;
  className?: string;
}

export default function OverallScore({ 
  score, 
  title = "Overall Confidence",
  className = ""
}: OverallScoreProps) {
  // Handle NaN/undefined values
  const normalizedScore = isNaN(score) || score === undefined ? 0 : Math.max(0, Math.min(100, score));
  
  // Color based on prediction confidence
  const getColorAndLabel = (score: number) => {
    if (score >= 80) return { color: "#16a34a", label: "Very High" };
    if (score >= 60) return { color: "#65a30d", label: "High" };
    if (score >= 40) return { color: "#d97706", label: "Medium" };
    if (score >= 20) return { color: "#ea580c", label: "Low" };
    return { color: "#dc2626", label: "Very Low" };
  };

  const { color, label } = getColorAndLabel(normalizedScore);

  // Check if this is a small inline version (for QA cards)
  const isSmall = className.includes("w-12") || className.includes("h-12");

  if (isSmall) {
    return (
      <div className={`${className}`} title={`${title}: ${label} (${normalizedScore.toFixed(0)}%)`}>
        <CircularProgressbar
          value={normalizedScore}
          text={`${normalizedScore.toFixed(0)}%`}
          styles={buildStyles({
            pathColor: color,
            textColor: color,
            trailColor: "#e5e7eb",
            pathTransitionDuration: 0.5,
            textSize: "24px",
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
            trailColor: "#e5e7eb",
            pathTransitionDuration: 0.5,
            textSize: "16px",
          })}
        />
      </div>

      <div className="w-full text-center space-y-2">
        <div className="text-xs font-medium text-gray-600 dark:text-gray-400">
          {title}
        </div>
        <div className="text-sm font-semibold" style={{ color }}>
          {label} ({score.toFixed(0)}%)
        </div>
        
        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 w-full mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-green-600 rounded-full shrink-0" />
            <span>Very High (≥80%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-lime-600 rounded-full shrink-0" />
            <span>High (60-79%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-amber-600 rounded-full shrink-0" />
            <span>Medium (40-59%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-orange-600 rounded-full shrink-0" />
            <span>Low (20-39%)</span>
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-red-600 rounded-full shrink-0" />
            <span>Very Low (&lt;20%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
