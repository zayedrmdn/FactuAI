"use client";

import { 
  DocumentIcon, 
  PhotoIcon, 
  VideoCameraIcon,
  PlayIcon,
  XMarkIcon,
  ChevronDownIcon,
  ChevronRightIcon
} from "@heroicons/react/24/outline";
import { HistoryItem as HistoryItemType } from "../../types/factcheck";

interface HistoryItemProps {
  item: HistoryItemType;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  onLoad: () => void;
  onDelete: () => void;
}

const VERDICT_ICONS = {
  true: { icon: "✅", color: "text-green-600" },
  mostly_true: { icon: "✅", color: "text-green-600" },
  false: { icon: "❌", color: "text-red-600" },
  mostly_false: { icon: "❌", color: "text-red-600" },
  half_true: { icon: "⚠️", color: "text-yellow-600" },
  barely_true: { icon: "⚠️", color: "text-yellow-600" },
  unknown: { icon: "❔", color: "text-gray-600" }
};

export default function HistoryItem({
  item,
  isExpanded = false,
  onToggleExpanded,
  onLoad,
  onDelete,
}: HistoryItemProps) {
  
  const getInputTypeIcon = () => {
    switch (item.type) {
      case "image":
        return <PhotoIcon className="w-4 h-4 text-blue-600" />;
      case "video":
        return <VideoCameraIcon className="w-4 h-4 text-purple-600" />;
      default:
        return <DocumentIcon className="w-4 h-4 text-gray-600" />;
    }
  };

  const getOverallAssessment = () => {
    if (!item.results?.length) return VERDICT_ICONS.unknown;
    
    // Count verdict types
    const counts = item.results.reduce((acc, r) => {
      acc[r.verdict] = (acc[r.verdict] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Find most severe verdict
    const severity = ["false", "mostly_false", "barely_true", "half_true", "mostly_true", "true"];
    for (const verdict of severity) {
      if (counts[verdict]) {
        return VERDICT_ICONS[verdict as keyof typeof VERDICT_ICONS] || VERDICT_ICONS.unknown;
      }
    }
    
    return VERDICT_ICONS.unknown;
  };

  const calculateAverageConfidence = () => {
    if (!item.results?.length) return 0;
    const total = item.results.reduce((sum, r) => sum + (r.confidence || 0), 0);
    return (total / item.results.length) * 100;
  };

  const overallAssessment = getOverallAssessment();
  const avgConfidence = calculateAverageConfidence();
  const hasResults = item.results && Array.isArray(item.results) && item.results.length > 0;

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors group">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3 flex-1">
          <div className="flex items-center gap-2 flex-shrink-0">
            {getInputTypeIcon()}
            {hasResults && (
              <span className="text-lg" title={`Overall assessment: ${overallAssessment.color}`}>
                {overallAssessment.icon}
              </span>
            )}
          </div>
          
          <div className="flex flex-col min-w-0 flex-1">
            <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {item.type === "image" ? "Image Analysis" :
               item.type === "video" ? "Video Analysis" :
               "Text Analysis"}
              {!hasResults && (
                <span className="ml-2 text-xs text-gray-500">(Extracted text only)</span>
              )}
            </div>
            
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {hasResults ? (
                <>
                  {item.results?.length || 0} claim{(item.results?.length || 0) !== 1 ? 's' : ''}
                  {avgConfidence > 0 && (
                    <span className="ml-1">
                      • {avgConfidence.toFixed(0)}% avg confidence
                    </span>
                  )}
                </>
              ) : (
                "Text extracted • Ready for fact-checking"
              )}
            </div>
            
            <div className="text-xs text-gray-400 mt-1">
              {formatTimestamp(item.timestamp)}
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2 flex-shrink-0">
          {onToggleExpanded && (
            <button
              onClick={onToggleExpanded}
              className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? (
                <ChevronDownIcon className="w-4 h-4" />
              ) : (
                <ChevronRightIcon className="w-4 h-4" />
              )}
            </button>
          )}
          
          <button
            onClick={onDelete}
            className="p-1 text-gray-400 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
            title="Delete"
          >
            <XMarkIcon className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Preview Text */}
      <div className="mb-3">
        <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
          {item.input.length > 150 ? `${item.input.substring(0, 150)}...` : item.input}
        </p>
      </div>

      {/* Metadata for specific types */}
      {item.metadata && (
        <div className="mb-3 text-xs text-gray-500">
          {item.type === "image" && item.metadata.aiScore !== undefined && (
            <div className="flex items-center gap-2">
              <span>AI Detection: {item.metadata.aiScore.toFixed(1)}%</span>
            </div>
          )}
          {item.type === "video" && item.metadata.filename && (
            <div className="flex items-center gap-2">
              <VideoCameraIcon className="w-3 h-3" />
              <span>{item.metadata.filename}</span>
            </div>
          )}
        </div>
      )}

      {/* Expanded Content */}
      {isExpanded && hasResults && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-3">
          <div className="text-sm">
            <h4 className="font-medium mb-2 text-gray-900 dark:text-gray-100">Summary</h4>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              {item.summary}
            </p>
          </div>
          
          <div className="text-sm">
            <h4 className="font-medium mb-2 text-gray-900 dark:text-gray-100">
              Claims ({item.results?.length || 0})
            </h4>
            <div className="space-y-2">
              {(item.results || []).slice(0, 3).map((result, index) => {
                const verdictInfo = VERDICT_ICONS[result.verdict as keyof typeof VERDICT_ICONS] || VERDICT_ICONS.unknown;
                return (
                  <div key={index} className="flex items-start gap-2 text-xs">
                    <span className="flex-shrink-0 mt-0.5">{verdictInfo.icon}</span>
                    <span className="text-gray-600 dark:text-gray-300 line-clamp-2">
                      {result.claim}
                    </span>
                  </div>
                );
              })}
              {(item.results?.length || 0) > 3 && (
                <div className="text-xs text-gray-500">
                  ... and {(item.results?.length || 0) - 3} more claim{((item.results?.length || 0) - 3) !== 1 ? 's' : ''}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2 mt-4">
        <button
          onClick={onLoad}
          className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm px-3 py-2 rounded-lg transition-colors font-medium"
        >
          Load Results
        </button>
        
        {item.type === "video" && item.metadata?.videoUrl && (
          <button
            onClick={() => {
              // This could open a video modal or navigate to video view
              console.log("Play video:", item.metadata?.videoUrl);
            }}
            className="flex items-center gap-1 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm px-3 py-2 rounded-lg transition-colors"
            title="Play video"
          >
            <PlayIcon className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
