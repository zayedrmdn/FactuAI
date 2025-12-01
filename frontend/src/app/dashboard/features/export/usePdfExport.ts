"use client";

import { useCallback } from "react";
import { toast } from "sonner";
import { exportToPdf } from "./pdfExport";
import { FactCheckResult, QAResult } from "../../types/factcheck";

type CombinedResult = FactCheckResult | QAResult;

interface UsePdfExportProps {
  results: CombinedResult[];
  summary: string;
  averageConfidence: number;
  aiScore?: number | null;
  isQAOnly: boolean;
}

/**
 * Hook for handling PDF export functionality
 */
export function usePdfExport({
  results,
  summary,
  averageConfidence,
  aiScore,
  isQAOnly
}: UsePdfExportProps) {
  
  const handleExportPdf = useCallback(() => {
    try {
      exportToPdf({
        results,
        summary,
        averageConfidence,
        aiScore,
        isQAOnly
      });
      
      toast.success("PDF exported successfully");
    } catch (err) {
      console.error("PDF export error:", err);
      
      if (err instanceof Error) {
        toast.error(err.message);
      } else {
        toast.error("Failed to export PDF");
      }
    }
  }, [results, summary, averageConfidence, aiScore, isQAOnly]);
  
  return {
    exportPdf: handleExportPdf
  };
}
