import { jsPDF } from 'jspdf';
import { FactCheckResult, QAResult } from '@/types/dashboard/factcheck';

type CombinedResult = FactCheckResult | QAResult;

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

// Professional color palette
const COLORS = {
  primary: [30, 64, 175] as [number, number, number],
  text: [31, 41, 55] as [number, number, number],
  muted: [107, 114, 128] as [number, number, number],
  border: [229, 231, 235] as [number, number, number],
  success: [22, 163, 74] as [number, number, number],
  danger: [220, 38, 38] as [number, number, number],
  warning: [217, 119, 6] as [number, number, number],
  bgLight: [249, 250, 251] as [number, number, number],
};

// Typography constants for consistent spacing
const LINE_HEIGHT = {
  body: 14,
  small: 11,
  tiny: 10,
};

const FONT_SIZE = {
  title: 22,
  subtitle: 12,
  sectionHeader: 12,
  body: 10,
  small: 9,
  tiny: 8,
};

// Maximum items to show to prevent extremely long PDFs
const LIMITS = {
  maxEvidence: 3,
  maxSources: 10,
  urlMaxLength: 90,
};

/**
 * Check if we need a new page, and add one if necessary
 */
function checkPageBreak(
  doc: jsPDF,
  y: number,
  pageHeight: number,
  requiredSpace: number,
  margin: number
): number {
  if (y + requiredSpace > pageHeight - 50) {
    doc.addPage();
    return margin + 20;
  }
  return y;
}

/**
 * Renders the professional header
 */
function renderHeader(doc: jsPDF, pageWidth: number, margin: number): number {
  doc.setFillColor(...COLORS.primary);
  doc.rect(0, 0, pageWidth, 4, 'F');

  doc.setFontSize(FONT_SIZE.title);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.primary);
  doc.text('FactuAI', margin, 45);

  doc.setFontSize(FONT_SIZE.subtitle);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.muted);
  doc.text('Analysis Report', margin + 100, 45);

  doc.setFontSize(FONT_SIZE.small);
  const dateStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  doc.text(dateStr, pageWidth - margin, 45, { align: 'right' });

  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.5);
  doc.line(margin, 58, pageWidth - margin, 58);

  return 80;
}

/**
 * Renders the executive summary section
 */
function renderSummary(
  doc: jsPDF,
  summary: string,
  margin: number,
  maxWidth: number,
  pageHeight: number,
  startY: number
): number {
  if (!summary) return startY;

  let y = startY;

  doc.setFontSize(FONT_SIZE.sectionHeader);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  doc.text('EXECUTIVE SUMMARY', margin, y);
  y += 16;

  doc.setFillColor(...COLORS.bgLight);
  const summaryLines = doc.splitTextToSize(summary, maxWidth - 20);
  const boxHeight = summaryLines.length * LINE_HEIGHT.body + 20;

  y = checkPageBreak(doc, y, pageHeight, boxHeight + 10, margin);

  doc.roundedRect(margin, y - 6, maxWidth, boxHeight, 4, 4, 'F');

  doc.setFontSize(FONT_SIZE.body);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.text);
  doc.text(summaryLines, margin + 10, y + 8);

  return y + boxHeight + 20;
}

/**
 * Renders the metrics overview
 */
function renderMetrics(
  doc: jsPDF,
  averageConfidence: number,
  aiScore: number | null | undefined,
  results: CombinedResult[],
  pageWidth: number,
  margin: number,
  maxWidth: number,
  pageHeight: number,
  startY: number
): number {
  let y = checkPageBreak(doc, startY, pageHeight, 90, margin);

  doc.setFontSize(FONT_SIZE.sectionHeader);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  doc.text('OVERVIEW', margin, y);
  y += 18;

  const stats = results.reduce(
    (acc, r) => {
      if (!isQAResult(r)) {
        const label = r.label?.toLowerCase() || 'unknown';
        if (['true', 'mostly_true'].includes(label)) acc.verified++;
        else if (['false', 'mostly_false'].includes(label)) acc.false++;
        else acc.unclear++;
      }
      return acc;
    },
    { verified: 0, false: 0, unclear: 0 }
  );

  const metrics: { label: string; value: string; color: [number, number, number] }[] = [
    { label: 'Trust Score', value: `${averageConfidence.toFixed(0)}%`, color: COLORS.primary },
    { label: 'Claims', value: `${results.length}`, color: COLORS.text },
    { label: 'Verified', value: `${stats.verified}`, color: COLORS.success },
    { label: 'False', value: `${stats.false}`, color: COLORS.danger },
  ];

  if (aiScore !== undefined && aiScore !== null) {
    metrics.push({ label: 'AI Score', value: `${aiScore.toFixed(0)}%`, color: COLORS.warning });
  }

  const boxWidth = (maxWidth - (metrics.length - 1) * 10) / metrics.length;
  metrics.forEach((metric, i) => {
    const x = margin + i * (boxWidth + 10);

    doc.setFillColor(...COLORS.bgLight);
    doc.roundedRect(x, y, boxWidth, 45, 3, 3, 'F');

    doc.setFontSize(18);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...metric.color);
    doc.text(metric.value, x + boxWidth / 2, y + 22, { align: 'center' });

    doc.setFontSize(FONT_SIZE.tiny);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.muted);
    doc.text(metric.label.toUpperCase(), x + boxWidth / 2, y + 38, { align: 'center' });
  });

  return y + 65;
}

/**
 * Gets verdict styling
 */
function getVerdictStyle(label: string | undefined): {
  color: [number, number, number];
  text: string;
} {
  const normalized = label?.toLowerCase();
  if (normalized === 'true' || normalized === 'mostly_true') {
    return { color: COLORS.success, text: 'VERIFIED' };
  }
  if (normalized === 'false' || normalized === 'mostly_false') {
    return { color: COLORS.danger, text: 'FALSE' };
  }
  return { color: COLORS.warning, text: 'UNVERIFIED' };
}

/**
 * Renders a single claim result with robust pagination
 */
function renderClaimResult(
  doc: jsPDF,
  result: FactCheckResult,
  idx: number,
  margin: number,
  maxWidth: number,
  pageWidth: number,
  pageHeight: number,
  startY: number
): number {
  let y = startY;
  const verdictStyle = getVerdictStyle(result.label);

  // Claim header
  y = checkPageBreak(doc, y, pageHeight, 60, margin);

  doc.setFontSize(FONT_SIZE.body);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.muted);
  doc.text(`CLAIM ${idx + 1}`, margin, y);

  doc.setTextColor(...verdictStyle.color);
  doc.text(verdictStyle.text, pageWidth - margin, y, { align: 'right' });
  y += 12;

  if (result.confidence !== undefined) {
    doc.setFontSize(FONT_SIZE.tiny);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.muted);
    doc.text(`Confidence: ${(result.confidence * 100).toFixed(0)}%`, pageWidth - margin, y, {
      align: 'right',
    });
  }
  y += 6;

  // Claim text
  doc.setFontSize(FONT_SIZE.body);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  const claimLines = doc.splitTextToSize(result.claim, maxWidth);
  y = checkPageBreak(doc, y, pageHeight, claimLines.length * LINE_HEIGHT.body + 10, margin);
  doc.text(claimLines, margin, y);
  y += claimLines.length * LINE_HEIGHT.body + 8;

  // Analysis/Reasoning
  if (result.reasoning) {
    y = checkPageBreak(doc, y, pageHeight, 40, margin);

    doc.setFontSize(FONT_SIZE.small);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.muted);
    doc.text('Analysis:', margin, y);
    y += LINE_HEIGHT.small;

    // Normalize text to prevent weird character spacing issues
    const cleanReasoning = result.reasoning
      .replace(/[\u2018\u2019]/g, "'") // Smart single quotes to apostrophe
      .replace(/[\u201C\u201D]/g, '"') // Smart double quotes
      .replace(/\u00AD/g, '') // Remove soft hyphens (invisible hyphenation hints)
      .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015]/g, '-') // Normalize all dash variants to ASCII hyphen
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();

    // Set font to normal for the analysis text content
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.text);
    doc.setFontSize(FONT_SIZE.small);

    const reasoningLines = doc.splitTextToSize(cleanReasoning, maxWidth - 10);

    // Render reasoning line by line with page breaks
    for (const line of reasoningLines) {
      y = checkPageBreak(doc, y, pageHeight, LINE_HEIGHT.small + 5, margin);
      doc.text(line, margin, y);
      y += LINE_HEIGHT.small;
    }
    y += 6;
  }

  // Key Evidence (limit to prevent overflow)
  if (result.source_quotes && result.source_quotes.length > 0) {
    y = checkPageBreak(doc, y, pageHeight, 30, margin);

    doc.setFontSize(FONT_SIZE.small);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.muted);
    doc.text('Key Evidence:', margin, y);
    y += LINE_HEIGHT.small + 2;

    const evidenceToShow = result.source_quotes.slice(0, LIMITS.maxEvidence);
    for (const sq of evidenceToShow) {
      y = checkPageBreak(doc, y, pageHeight, 35, margin);

      doc.setFont('helvetica', 'italic');
      doc.setTextColor(...COLORS.text);
      const quoteText = `"${sq.quote}"`;
      const quoteLines = doc.splitTextToSize(quoteText, maxWidth - 12);
      doc.text(quoteLines, margin + 6, y);
      y += quoteLines.length * LINE_HEIGHT.small + 2;

      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...COLORS.muted);
      doc.text(`— ${sq.source}`, margin + 6, y);
      y += LINE_HEIGHT.small + 4;
    }

    if (result.source_quotes.length > LIMITS.maxEvidence) {
      doc.setTextColor(...COLORS.muted);
      doc.text(
        `(${result.source_quotes.length - LIMITS.maxEvidence} more evidence items not shown)`,
        margin + 6,
        y
      );
      y += LINE_HEIGHT.small;
    }
  }

  // Sources (limit to prevent extremely long PDFs)
  if (result.sources && result.sources.length > 0) {
    y = checkPageBreak(doc, y, pageHeight, 30, margin);

    doc.setFontSize(FONT_SIZE.small);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.muted);
    doc.text('Sources:', margin, y);
    y += LINE_HEIGHT.small;

    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.primary);

    const sourcesToShow = result.sources.slice(0, LIMITS.maxSources);
    for (let i = 0; i < sourcesToShow.length; i++) {
      y = checkPageBreak(doc, y, pageHeight, LINE_HEIGHT.small + 3, margin);

      const source = sourcesToShow[i];
      const sourceText = typeof source === 'string' ? source : String(source ?? '');
      const truncated =
        sourceText.length > LIMITS.urlMaxLength
          ? sourceText.substring(0, LIMITS.urlMaxLength - 3) + '...'
          : sourceText;
      doc.text(`${i + 1}. ${truncated}`, margin + 6, y);
      y += LINE_HEIGHT.small;
    }

    if (result.sources.length > LIMITS.maxSources) {
      doc.setTextColor(...COLORS.muted);
      doc.text(`+ ${result.sources.length - LIMITS.maxSources} more sources`, margin + 6, y);
      y += LINE_HEIGHT.small;
    }
  }

  // Divider
  y += 8;
  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.5);
  doc.line(margin, y, pageWidth - margin, y);

  return y + 15;
}

/**
 * Renders a QA result
 */
function renderQAResult(
  doc: jsPDF,
  result: QAResult,
  idx: number,
  margin: number,
  maxWidth: number,
  pageWidth: number,
  pageHeight: number,
  startY: number
): number {
  let y = checkPageBreak(doc, startY, pageHeight, 50, margin);

  doc.setFontSize(FONT_SIZE.body);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.primary);
  doc.text(`Q${idx + 1}:`, margin, y);

  doc.setTextColor(...COLORS.text);
  const questionLines = doc.splitTextToSize(result.question, maxWidth - 25);
  doc.text(questionLines, margin + 22, y);
  y += questionLines.length * LINE_HEIGHT.body + 8;

  doc.setFontSize(FONT_SIZE.body);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.text);
  const answerLines = doc.splitTextToSize(result.answer, maxWidth);

  for (const line of answerLines) {
    y = checkPageBreak(doc, y, pageHeight, LINE_HEIGHT.body + 3, margin);
    doc.text(line, margin, y);
    y += LINE_HEIGHT.body;
  }
  y += 10;

  doc.setDrawColor(...COLORS.border);
  doc.line(margin, y, pageWidth - margin, y);

  return y + 15;
}

/**
 * Renders all results with proper pagination
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
): void {
  if (results.length === 0) return;

  let y = checkPageBreak(doc, startY, pageHeight, 40, margin);

  doc.setFontSize(FONT_SIZE.sectionHeader);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  doc.text(isQAOnly ? 'QUESTIONS & ANSWERS' : 'DETAILED ANALYSIS', margin, y);
  y += 20;

  results.forEach((result, idx) => {
    if (isQAResult(result)) {
      y = renderQAResult(doc, result, idx, margin, maxWidth, pageWidth, pageHeight, y);
    } else {
      y = renderClaimResult(doc, result, idx, margin, maxWidth, pageWidth, pageHeight, y);
    }
  });
}

/**
 * Renders footer on all pages
 */
function renderFooter(doc: jsPDF, pageWidth: number, pageHeight: number, margin: number): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const totalPages = (doc as any).internal.getNumberOfPages();

  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);

    doc.setDrawColor(...COLORS.border);
    doc.setLineWidth(0.5);
    doc.line(margin, pageHeight - 35, pageWidth - margin, pageHeight - 35);

    doc.setFontSize(FONT_SIZE.tiny);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.muted);
    doc.text('Generated by FactuAI — AI-Powered Fact-Checking', margin, pageHeight - 22);
    doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 22, { align: 'right' });
  }
}

/**
 * Main export function - handles all edge cases
 */
export function exportToPdf(options: PdfExportOptions): void {
  const { results, summary, averageConfidence, aiScore, isQAOnly } = options;

  if (!results.length && !summary) {
    throw new Error('Nothing to export');
  }

  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 45;
  const maxWidth = pageWidth - 2 * margin;

  let y = renderHeader(doc, pageWidth, margin);
  y = renderSummary(doc, summary, margin, maxWidth, pageHeight, y);
  y = renderMetrics(
    doc,
    averageConfidence,
    aiScore,
    results,
    pageWidth,
    margin,
    maxWidth,
    pageHeight,
    y
  );
  renderResults(doc, results, isQAOnly, pageWidth, pageHeight, margin, maxWidth, y);

  renderFooter(doc, pageWidth, pageHeight, margin);

  const filename = `FactuAI-Report-${new Date().toISOString().split('T')[0]}.pdf`;
  doc.save(filename);
}
