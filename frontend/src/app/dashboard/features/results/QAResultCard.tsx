"use client";

import { QAResult } from "../../types/factcheck";
import { TextSize } from "../../types/ui";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionItem } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

export function QAResultCard({
  result,
  index,
  textSize,
  animationDelay
}: Readonly<{
  result: QAResult;
  index: number;
  textSize: TextSize;
  animationDelay: number;
}>) {
  const { question, answer, sources, confidence } = result;
  
  // Ensure confidence is a valid number
  const safeConfidence = typeof confidence === 'number' && !Number.isNaN(confidence) ? confidence : 0.8;

  // Calculate progress color
  const getProgressColor = (score: number) => {
    if (score >= 0.8) return "oklch(0.623 0.214 163.525)"; // emerald-500
    if (score >= 0.5) return "oklch(0.769 0.188 70.08)"; // amber-500
    return "oklch(0.627 0.265 303.9)"; // purple/indigo or red
  };

  const textSizeClass = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  }[textSize];

  return (
    <Card 
      className="overflow-hidden animate-in slide-in-from-left duration-300 border-l-4 border-l-primary"
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <Badge variant="secondary" className="mb-2">Question {index + 1}</Badge>
            <CardTitle className="text-lg font-medium leading-relaxed">
              {question}
            </CardTitle>
          </div>
          <div className="flex flex-col items-end gap-1 min-w-[100px]">
            <span className="text-xs font-medium text-muted-foreground">Confidence</span>
            <div className="flex items-center gap-2 w-full">
              <Progress value={safeConfidence * 100} className="h-2" indicatorColor={getProgressColor(safeConfidence)} />
              <span className="text-xs font-bold">{(safeConfidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className={cn("text-foreground leading-relaxed", textSizeClass)}>
          {answer}
        </div>

        {sources.length > 0 && (
          <Accordion>
            <AccordionItem 
              title={
                <div className="flex items-center gap-2">
                  <ExternalLink className="h-4 w-4" />
                  <span>Sources ({sources.length})</span>
                </div>
              }
            >
              <ul className="space-y-2 pt-2">
                {sources.map((url, i) => (
                  <li key={`source-${url.slice(0, 50)}`} className="flex items-start gap-2 text-sm">
                    <span className="mt-1 text-xs text-muted-foreground">{i + 1}.</span>
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
          </Accordion>
        )}
      </CardContent>
    </Card>
  );
}