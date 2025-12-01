"use client";

import OverallScore from "@/components/ui/OverallScore";
import { QAResult } from "../../types/factcheck";
import { TextSize } from "../../types/ui";

export function QAResultCard({
  result,
  index,
  textSize,
  animationDelay
}: {
  result: QAResult;
  index: number;
  textSize: TextSize;
  animationDelay: number;
}) {
  const { question, answer, sources, confidence } = result;
  
  // Ensure confidence is a valid number
  const safeConfidence = typeof confidence === 'number' && !isNaN(confidence) ? confidence : 0.8;

  return (
    <div
      style={{ animationDelay: `${animationDelay}ms` }}
      className="p-6 mb-4 bg-white dark:bg-gray-800 rounded-lg border animate-in"
    >
      {/* header: question + circular confidence */}
      <div className="flex items-start justify-between">
        <h3 className="font-semibold text-base md:text-lg">
          Q{index + 1}: {question}
        </h3>
        <div className="w-12 h-12">
          {/* OverallScore expects 0–100 */}
          <OverallScore
            score={safeConfidence * 100}
            title="Confidence"
            className="w-12 h-12"
          />
        </div>
      </div>

      {/* answer text */}
      <p
        className={`mt-4 text-gray-900 dark:text-gray-100 leading-relaxed ${
          textSize === "sm" ? "text-sm" :
          textSize === "lg" ? "text-lg" : "text-base"
        }`}
      >
        {answer}
      </p>

      {/* sources list */}
      {sources.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Sources
          </h4>
          <ul className="mt-2 space-y-1 pl-4 list-disc text-sm text-blue-600 dark:text-blue-400">
            {sources.map((url, i) => (
              <li key={i}>
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline break-all"
                >
                  {new URL(url).hostname}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}