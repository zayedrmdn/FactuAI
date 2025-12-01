import { jsPDF } from "jspdf";
import { FactCheckResult, QAResult } from "../../types/factcheck";

type CombinedResult = FactCheckResult | QAResult;

/** True if this is a QAResult, false otherwise */
function isQAResult(r: CombinedResult): r is QAResult {
  return (r as QAResult).answer !== undefined;
}

interface PdfExportOptions {
  results: CombinedResult[];
  summary: string;
  averageConfidence: number;
  aiScore?: number | null;
  isQAOnly: boolean;
}

/**
 * Helper function to detect URLs in text and make them clickable in PDF
 */
function addTextWithLinks(
  doc: jsPDF,
  text: string,
  x: number,
  y: number,
  maxWidth: number
): number {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const parts = text.split(urlRegex);
  let currentX = x;
  let currentY = y;
  
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (urlRegex.test(part)) {
      // This is a URL - make it blue and clickable
      doc.setTextColor(0, 0, 255); // Blue color
      const lines = doc.splitTextToSize(part, maxWidth);
      doc.text(lines, currentX, currentY);
      // Add link annotation
      doc.link(currentX, currentY - 10, doc.getTextWidth(lines[0]), 12, { url: part });
      doc.setTextColor(0, 0, 0); // Reset to black
      currentY += lines.length * 12;
    } else if (part.trim()) {
      // Regular text
      const lines = doc.splitTextToSize(part, maxWidth);
      doc.text(lines, currentX, currentY);
      currentY += lines.length * 12;
    }
  }
  return currentY;
}

/**
 * Renders the header section of the PDF
 */
function renderHeader(doc: jsPDF, pageWidth: number, margin: number): number {
  // Header section with better styling
  doc.setFillColor(240, 248, 255); // Light blue background
  doc.rect(0, 0, pageWidth, 90, 'F');
  
  doc.setFontSize(20);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(30, 64, 175); // Dark blue
  doc.text("FactuAI Analysis Report", margin, 40);
  
  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(75, 85, 99); // Gray
  doc.text(`Generated on ${new Date().toLocaleString()}`, margin, 65);
  
  doc.setTextColor(0, 0, 0); // Reset to black
  return 120; // Return starting Y position for content
}

/**
 * Renders the summary section of the PDF
 */
function renderSummary(
  doc: jsPDF,
  summary: string,
  pageWidth: number,
  margin: number,
  maxWidth: number,
  startY: number
): number {
  let y = startY;
  
  if (!summary) return y;
  
  // Add separator line
  doc.setDrawColor(229, 231, 235);
  doc.setLineWidth(1);
  doc.line(margin, y - 10, pageWidth - margin, y - 10);
  
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(75, 85, 99);
  doc.text("Executive Summary", margin, y);
  y += 25;
  
  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(0, 0, 0);
  
  // Add background for summary
  const summaryLines = doc.splitTextToSize(summary, maxWidth - 20);
  const summaryHeight = summaryLines.length * 14 + 20;
  doc.setFillColor(249, 250, 251);
  doc.roundedRect(margin, y - 5, maxWidth, summaryHeight, 3, 3, 'F');
  
  doc.text(summaryLines, margin + 10, y + 10);
  y += summaryHeight + 15;
  
  return y;
}

/**
 * Renders the metrics section (confidence and AI detection)
 */
function renderMetrics(
  doc: jsPDF,
  averageConfidence: number,
  aiScore: number | null | undefined,
  pageWidth: number,
  margin: number,
  startY: number
): number {
  let y = startY;
  
  if (averageConfidence <= 0 && (aiScore === undefined || aiScore === null)) {
    return y;
  }
  
  doc.setDrawColor(229, 231, 235);
  doc.line(margin, y - 5, pageWidth - margin, y - 5);
  
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(75, 85, 99);
  doc.text("Analysis Metrics", margin, y + 10);
  y += 35;

  if (averageConfidence > 0) {
    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(34, 197, 94); // Green
    doc.text(`Overall Confidence: ${averageConfidence.toFixed(1)}%`, margin, y);
    y += 20;
  }

  if (aiScore !== undefined && aiScore !== null) {
    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(168, 85, 247); // Purple
    doc.text(`AI Content Detection: ${aiScore.toFixed(1)}%`, margin, y);
    y += 25;
  }
  
  doc.setTextColor(0, 0, 0); // Reset color
  return y;
}

/**
 * Renders a QA result card
 */
function renderQAResult(
  doc: jsPDF,
  result: QAResult,
  idx: number,
  margin: number,
  maxWidth: number,
  startY: number
): number {
  let y = startY;
  const cardStartY = y - 10;
  
  // QA Result styling
  doc.setFillColor(254, 249, 195); // Light yellow
  doc.roundedRect(margin, cardStartY, maxWidth, 20, 3, 3, 'F');
  
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(146, 64, 14); // Orange-brown
  doc.text(`Q${idx + 1}: ${result.question}`, margin + 10, y + 5);
  y += 25;
  
  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(0, 0, 0);
  const answerLines = doc.splitTextToSize(result.answer, maxWidth - 20);
  doc.text(answerLines, margin + 10, y);
  y += answerLines.length * 14 + 15;

  if (result.sources?.length > 0) {
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(75, 85, 99);
    doc.text("Sources:", margin + 10, y);
    y += 15;
    
    doc.setFont('helvetica', 'normal');
    result.sources.forEach((source, sourceIdx) => {
      y = addTextWithLinks(doc, `${sourceIdx + 1}. ${source}`, margin + 20, y, maxWidth - 30);
      y += 5;
    });
    y += 10;
  }
  
  return y;
}

/**
 * Renders a fact-check result card
 */
function renderFactCheckResult(
  doc: jsPDF,
  result: FactCheckResult,
  idx: number,
  margin: number,
  maxWidth: number,
  startY: number
): number {
  let y = startY;
  const cardStartY = y - 10;
  
  // Color code by verdict
  let bgColor: [number, number, number], textColor: [number, number, number];
  switch (result.label?.toLowerCase()) {
    case 'true':
    case 'mostly true':
      bgColor = [240, 253, 244]; // Light green
      textColor = [22, 163, 74]; // Green
      break;
    case 'false':
    case 'mostly false':
      bgColor = [254, 242, 242]; // Light red
      textColor = [220, 38, 38]; // Red
      break;
    default:
      bgColor = [255, 251, 235]; // Light yellow
      textColor = [217, 119, 6]; // Yellow
  }
  
  doc.setFillColor(bgColor[0], bgColor[1], bgColor[2]);
  
  // Calculate card height dynamically
  const claimLines = doc.splitTextToSize(result.claim, maxWidth - 20);
  let cardHeight = 60 + claimLines.length * 14;
  if (result.source_quotes?.length) cardHeight += result.source_quotes.length * 25;
  if (result.evidence?.length) cardHeight += 40;
  if (result.sources?.length) cardHeight += result.sources.length * 15 + 25;
  
  doc.roundedRect(margin, cardStartY, maxWidth, Math.min(cardHeight, 100), 3, 3, 'F');
  
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(textColor[0], textColor[1], textColor[2]);
  doc.text(`Claim ${idx + 1}:`, margin + 10, y + 5);
  y += 20;
  
  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(0, 0, 0);
  doc.text(claimLines, margin + 10, y);
  y += claimLines.length * 14 + 15;

  // Verdict with simple text
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(textColor[0], textColor[1], textColor[2]);
  doc.text(`Verdict: ${result.label}`, margin + 10, y);
  y += 15;

  if (result.confidence) {
    doc.setTextColor(75, 85, 99);
    doc.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, margin + 10, y);
    y += 15;
  }

  doc.setTextColor(0, 0, 0);

  // Evidence section
  if (result.source_quotes && result.source_quotes.length > 0) {
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(75, 85, 99);
    doc.text("Evidence:", margin + 10, y);
    y += 15;
    
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(55, 65, 81);
    result.source_quotes.forEach((quote) => {
      const quoteText = `"${quote.quote}" - ${quote.source}`;
      const quoteLines = doc.splitTextToSize(quoteText, maxWidth - 30);
      doc.text(quoteLines, margin + 20, y);
      y += quoteLines.length * 12 + 8;
    });
    y += 10;
  } else if (result.evidence && result.evidence.length > 0) {
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(75, 85, 99);
    doc.text("Evidence:", margin + 10, y);
    y += 15;
    
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(55, 65, 81);
    const evidenceText = Array.isArray(result.evidence) ? result.evidence.join('. ') : result.evidence;
    const evidenceLines = doc.splitTextToSize(evidenceText, maxWidth - 30);
    doc.text(evidenceLines, margin + 20, y);
    y += evidenceLines.length * 12 + 15;
  }

  // Sources
  if (result.sources?.length > 0) {
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(75, 85, 99);
    doc.text("Sources:", margin + 10, y);
    y += 15;
    
    doc.setFont('helvetica', 'normal');
    result.sources.forEach((source, sourceIdx) => {
      y = addTextWithLinks(doc, `${sourceIdx + 1}. ${source}`, margin + 20, y, maxWidth - 30);
      y += 5;
    });
    y += 10;
  }
  
  return y;
}

/**
 * Renders the results section
 */
function renderResults(
  doc: jsPDF,
  results: CombinedResult[],
  isQAOnly: boolean,
  pageWidth: number,
  pageHeight: number,
  margin: number,
  maxWidth: number,
  startY: number
): number {
  let y = startY;
  
  if (results.length === 0) return y;
  
  doc.setDrawColor(229, 231, 235);
  doc.line(margin, y - 5, pageWidth - margin, y - 5);
  
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(75, 85, 99);
  const sectionTitle = isQAOnly ? "Questions & Answers" : "Fact-Check Results";
  doc.text(sectionTitle, margin, y + 10);
  y += 35;

  results.forEach((result, idx) => {
    // Check if we need a new page (with more space buffer)
    if (y > pageHeight - 150) {
      doc.addPage();
      y = 50;
    }

    if (isQAResult(result)) {
      y = renderQAResult(doc, result, idx, margin, maxWidth, y);
    } else {
      y = renderFactCheckResult(doc, result, idx, margin, maxWidth, y);
    }

    y += 25; // Space between results
  });
  
  return y;
}

/**
 * Renders the footer on all pages
 */
function renderFooter(doc: jsPDF, pageHeight: number, margin: number): void {
  const totalPages = (doc as any).internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(156, 163, 175);
    doc.text(`Generated by FactuAI - Page ${i} of ${totalPages}`, margin, pageHeight - 20);
  }
}

/**
 * Main function to export results to PDF
 */
export function exportToPdf(options: PdfExportOptions): void {
  const { results, summary, averageConfidence, aiScore, isQAOnly } = options;
  
  if (!results.length && !summary) {
    throw new Error("Nothing to export");
  }
  
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 50;
  const maxWidth = pageWidth - 2 * margin;
  
  // Render sections
  let y = renderHeader(doc, pageWidth, margin);
  y = renderSummary(doc, summary, pageWidth, margin, maxWidth, y);
  y = renderMetrics(doc, averageConfidence, aiScore, pageWidth, margin, y);
  y = renderResults(doc, results, isQAOnly, pageWidth, pageHeight, margin, maxWidth, y);
  
  // Add footer to all pages
  renderFooter(doc, pageHeight, margin);
  
  // Save the PDF
  const filename = `FactuAI-Report-${new Date().toISOString().split('T')[0]}.pdf`;
  doc.save(filename);
}
