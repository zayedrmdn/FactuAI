"use client";

import { TextSize } from "../../types/ui";
import { FactCheckResult } from "../../types/factcheck";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionItem } from "@/components/ui/accordion";
import { CheckCircle2, AlertTriangle, XCircle, HelpCircle, ExternalLink, Quote } from "lucide-react";
import { cn } from "@/lib/utils";

interface ClaimCardProps {
  result: FactCheckResult;
  index: number;
  textSize: TextSize;
  animationDelay: number;
}

const VERDICT_CONFIG: Record<string, { variant: "success" | "warning" | "destructive" | "secondary", label: string, icon: any }> = {
  true: { variant: "success", label: "True", icon: CheckCircle2 },
  mostly_true: { variant: "success", label: "Mostly True", icon: CheckCircle2 },
  half_true: { variant: "warning", label: "Half True", icon: AlertTriangle },
  barely_true: { variant: "warning", label: "Barely True", icon: AlertTriangle },
  false: { variant: "destructive", label: "False", icon: XCircle },
  mostly_false: { variant: "destructive", label: "Mostly False", icon: XCircle },
  unknown: { variant: "secondary", label: "Unknown", icon: HelpCircle },
};

export default function ClaimCard({ result, index, textSize, animationDelay }: ClaimCardProps) {
  const config = VERDICT_CONFIG[result.label] || VERDICT_CONFIG.unknown;
  const VerdictIcon = config.icon;

  const textSizeClass = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  }[textSize];

  // Calculate progress color based on confidence
  const getProgressColor = (score: number) => {
    if (score >= 0.8) return "oklch(0.623 0.214 163.525)"; // emerald-500
    if (score >= 0.5) return "oklch(0.769 0.188 70.08)"; // amber-500
    return "oklch(0.627 0.265 303.9)"; // purple/indigo or red
  };

  return (
    <Card 
      className="overflow-hidden animate-in slide-in-from-left duration-300 border-l-4"
      style={{ 
        animationDelay: `${animationDelay}ms`,
        borderLeftColor: config.variant === 'success' ? 'var(--emerald-500)' : 
                        config.variant === 'warning' ? 'var(--amber-500)' : 
                        config.variant === 'destructive' ? 'var(--destructive)' : 
                        'var(--secondary)'
      }}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-muted-foreground">Claim {index + 1}</Badge>
              <Badge variant={config.variant} className="gap-1">
                <VerdictIcon className="h-3 w-3" />
                {config.label}
              </Badge>
            </div>
            <CardTitle className={cn("font-medium leading-relaxed", textSizeClass)}>
              {result.claim}
            </CardTitle>
          </div>
          {result.confidence !== undefined && (
            <div className="flex flex-col items-end gap-1 min-w-[100px]">
              <span className="text-xs font-medium text-muted-foreground">Confidence</span>
              <div className="flex items-center gap-2 w-full">
                <Progress value={result.confidence * 100} className="h-2" indicatorColor={getProgressColor(result.confidence)} />
                <span className="text-xs font-bold">{(result.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Explanation */}
        {result.explanation && (
          <div className="rounded-md bg-muted/50 p-3 text-sm text-muted-foreground">
            {result.explanation}
          </div>
        )}

        {/* Evidence Accordion */}
        <Accordion>
          <AccordionItem 
            title={
              <div className="flex items-center gap-2">
                <Quote className="h-4 w-4" />
                <span>Evidence & Analysis</span>
              </div>
            }
          >
             {result.source_quotes && result.source_quotes.length > 0 ? (
              <div className="space-y-3 pt-2">
                {result.source_quotes.map((quote, idx) => (
                  <div key={idx} className="relative rounded-lg border bg-card p-4 shadow-sm">
                    <div className="absolute -left-3 top-4 rounded-full bg-primary p-1 text-primary-foreground shadow-sm">
                      <CheckCircle2 className="h-3 w-3" />
                    </div>
                    <blockquote className="border-l-2 border-primary/20 pl-4 text-sm italic text-muted-foreground">
                      "{quote.quote}"
                    </blockquote>
                    <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                      <span>— {quote.source}</span>
                      <a
                        href={quote.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-primary hover:underline"
                      >
                        Source <ExternalLink className="h-3 w-3" />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {result.evidence || "No specific evidence quotes available."}
              </p>
            )}
          </AccordionItem>

          {result.sources?.length > 0 && (
            <AccordionItem 
              title={
                <div className="flex items-center gap-2">
                  <ExternalLink className="h-4 w-4" />
                  <span>Sources ({result.sources.length})</span>
                </div>
              }
            >
              <ul className="space-y-2 pt-2">
                {result.sources.map((url, sourceIndex) => (
                  <li key={sourceIndex} className="flex items-start gap-2 text-sm">
                    <span className="mt-1 text-xs text-muted-foreground">{sourceIndex + 1}.</span>
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline break-all"
                    >
                      {url}
                    </a>
                  </li>
                ))}
              </ul>
            </AccordionItem>
          )}
        </Accordion>
      </CardContent>
    </Card>
  );
}