"use client";

import { TextSize } from "../../types/ui";
import { FactCheckResult } from "../../types/factcheck";

interface ClaimCardProps {
  result: FactCheckResult;
  index: number;
  textSize: TextSize;
  animationDelay: number;
}

const VERDICT_COLORS: Record<string, string> = {
  true: "#16a34a",
  mostly_true: "#16a34a",
  half_true: "#d97706",
  barely_true: "#d97706",
  false: "#dc2626",
  mostly_false: "#dc2626",
  unknown: "#6b7280",
};

const VERDICT_LABELS: Record<string, string> = {
  true: "True",
  mostly_true: "Mostly True",
  half_true: "Half True", 
  barely_true: "Barely True",
  false: "False",
  mostly_false: "Mostly False",
  unknown: "Unknown",
};

export default function ClaimCard({ result, index, textSize, animationDelay }: ClaimCardProps) {
  const verdictColor = VERDICT_COLORS[result.label] || VERDICT_COLORS.unknown;
  const verdictLabel = VERDICT_LABELS[result.label] || result.label;

  const textSizeClass = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  }[textSize];

  return (
    <div
      className="border dark:border-neutral-700 rounded-lg p-4 space-y-3 animate-in slide-in-from-left duration-300 bg-white dark:bg-neutral-800"
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-medium text-gray-900 dark:text-gray-100">
            Claim {index + 1}
          </h4>
          {result.confidence !== undefined && (
            <span className="text-xs text-muted-foreground bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          )}
        </div>

        <p className={`${textSizeClass} text-gray-800 dark:text-gray-200 mb-3`}>
          {result.claim}
        </p>

        <div className="mb-3">
          <span
            className="inline-block px-3 py-1 rounded-full text-white text-sm font-medium"
            style={{ backgroundColor: verdictColor }}
          >
            {verdictLabel}
          </span>
        </div>

        {result.source_quotes && result.source_quotes.length > 0 ? (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">Evidence</h5>
            <div className="space-y-3">
              {result.source_quotes.map((quote, idx) => (
                <div key={idx} className="bg-gray-50 dark:bg-gray-700 p-3 rounded border-l-4 border-green-500">
                  <div className="flex items-start gap-2">
                    <span className="text-green-600 dark:text-green-400 text-sm font-bold mt-0.5">✓</span>
                    <div className="flex-1">
                      <p className="text-sm text-gray-700 dark:text-gray-300 mb-2 italic">
                        "{quote.quote}"
                      </p>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-600 dark:text-gray-400">
                          — {quote.source}
                        </span>
                        <a
                          href={quote.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 hover:underline"
                        >
                          [source]
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : result.evidence && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">Evidence</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-3 rounded">
              {result.evidence}
            </p>
          </div>
        )}

        {result.explanation && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">Explanation</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {result.explanation}
            </p>
          </div>
        )}

        {result.sources?.length > 0 && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Sources ({result.sources.length})
            </h5>
            <ul className="space-y-1">
              {result.sources.map((url, sourceIndex) => (
                <li key={sourceIndex} className="flex items-start gap-2">
                  <span className="text-xs text-gray-400 mt-1">{sourceIndex + 1}.</span>
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 hover:underline break-all text-sm"
                  >
                    {url}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}