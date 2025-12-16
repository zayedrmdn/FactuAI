/**
 * Pipeline Model Configuration Panel
 *
 * Allows users to configure which AI model is used for each pipeline task:
 * - Intent Detection
 * - Claim Extraction
 * - Reasoning & Verification
 * - Summarization
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Settings, Brain, RotateCcw, ChevronDown, AlignLeft, Zap, FileText } from 'lucide-react';
import { useSystemConfig } from '@/hooks/useSystemConfig';
import {
  usePipelineModelsStore,
  getModelById,
  getModelsByProvider,
  modelRegistry,
} from '@/features/ai-providers';
import type { PipelineTask, AIProvider } from '@/features/ai-providers';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/primitives';

// Full Path: src/features/ai-providers/components/PipelineModelConfig.tsx
const taskConfig: Record<
  PipelineTask,
  {
    label: string;
    description: string;
    icon: React.ElementType;
    color: string;
  }
> = {
  intent: {
    label: 'Intent Detection',
    description: 'Fast model for routing (Factual vs General)',
    icon: Zap,
    color: 'text-amber-500',
  },
  extraction: {
    label: 'Extraction & Search Pivot',
    description: 'Extracts claims and evaluates search result relevance (Pivot)',
    icon: FileText,
    color: 'text-blue-500',
  },
  reasoning: {
    label: 'Reasoning & Verification',
    description: 'Model used for deep analysis, pivot decisions, and final verdict',
    icon: Brain,
    color: 'text-purple-500',
  },
  summary: {
    label: 'Summarization',
    description: 'Model used for generating the executive summary',
    icon: AlignLeft,
    color: 'text-green-500',
  },
};

export function PipelineModelConfig({
  compact = false,
  textSize = 'md',
}: {
  compact?: boolean;
  textSize?: 'sm' | 'md' | 'lg';
}) {
  const [isExpanded, setIsExpanded] = useState(!compact);
  const { reasoning, intent, extraction, summary, setTaskModel, resetToDefaults, syncWithBackend } =
    usePipelineModelsStore();
  const { config: systemConfig } = useSystemConfig();

  // Sync with backend defaults when config is available
  useEffect(() => {
    if (systemConfig?.models) {
      syncWithBackend({
        reasoning: systemConfig.models.default_reasoning,
        intent: systemConfig.models.default_intent,
      });
    }
  }, [systemConfig, syncWithBackend]);

  const taskSelections = { reasoning, intent, extraction, summary };
  const activeTask = usePipelineModelsStore((state) => state.activeTask);

  const labelClass = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }[textSize];

  const selectTriggerClass = {
    sm: 'h-7 text-[10px]',
    md: 'h-8 text-xs',
    lg: 'h-9 text-sm',
  }[textSize];

  const renderTask = (task: PipelineTask) => {
    const config = taskConfig[task];
    const selection = taskSelections[task];
    const currentModel = getModelById(selection.modelId);
    const isActive = activeTask === task;
    const Icon = config.icon;

    if (compact) {
      return (
        <div key={task} className="py-2 border-b last:border-0 bg-transparent group">
          <div className="flex items-center gap-2 mb-2">
            <div className={`p-1 rounded-md bg-muted/50 group-hover:bg-muted transition-colors`}>
              <Icon className={`h-3.5 w-3.5 ${config.color}`} />
            </div>
            <span className={`${labelClass} font-medium`}>{config.label}</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Select
              value={selection.provider}
              onValueChange={(provider) => {
                const providerModels = getModelsByProvider(provider);
                const recommendedModel = providerModels.find((m) => m.isRecommended);
                const defaultModel = recommendedModel || providerModels[0];
                if (defaultModel) {
                  setTaskModel(task, provider as AIProvider, defaultModel.id);
                }
              }}
            >
              <SelectTrigger className={selectTriggerClass}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {modelRegistry.providers.map((provider) => (
                  <SelectItem key={provider.id} value={provider.id} className="text-xs">
                    {provider.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={selection.modelId}
              onValueChange={(modelId) => setTaskModel(task, selection.provider, modelId)}
            >
              <SelectTrigger className={selectTriggerClass}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[300px]">
                {getModelsByProvider(selection.provider).map((model) => (
                  <SelectItem key={model.id} value={model.id} className="text-xs">
                    <div className="flex items-center gap-2 max-w-full">
                      <span className="truncate">{model.displayName}</span>
                      {model.isRecommended && (
                        <Badge variant="secondary" className="text-[10px] px-1 py-0 h-4">
                          Recommended
                        </Badge>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      );
    }

    return (
      <div
        key={task}
        className={`group space-y-3 p-3 rounded-lg border transition-all duration-200 ${
          isActive
            ? 'border-primary/50 bg-primary/5 ring-1 ring-primary/20'
            : 'border-transparent hover:border-border/50 hover:bg-muted/30'
        }`}
      >
        <div className="flex flex-col sm:flex-row items-start gap-3">
          <div
            className={`mt-1 p-2 rounded-lg bg-background border shadow-sm shrink-0 transition-colors group-hover:border-primary/20`}
          >
            <Icon className={`h-5 w-5 ${config.color}`} />
          </div>

          <div className="flex-1 w-full space-y-3">
            <div>
              <h4 className="text-sm sm:text-base font-medium text-foreground">{config.label}</h4>
              <p className="text-xs sm:text-sm text-muted-foreground">{config.description}</p>
            </div>

            {/* Responsive Grid for Provider & Model Selection */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3">
              {/* Provider Selection */}
              <div className="space-y-1.5">
                <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground/70 ml-1">
                  Provider
                </span>
                <Select
                  value={selection.provider}
                  onValueChange={(provider) => {
                    const providerModels = getModelsByProvider(provider);
                    const recommendedModel = providerModels.find((m) => m.isRecommended);
                    const defaultModel = recommendedModel || providerModels[0];
                    if (defaultModel) {
                      setTaskModel(task, provider as AIProvider, defaultModel.id);
                    }
                  }}
                >
                  <SelectTrigger className="h-9 text-xs sm:text-sm bg-background/50 hover:bg-background transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {modelRegistry.providers.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Model Selection */}
              <div className="space-y-1.5">
                <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground/70 ml-1">
                  Model
                </span>
                <Select
                  value={selection.modelId}
                  onValueChange={(modelId) => setTaskModel(task, selection.provider, modelId)}
                >
                  <SelectTrigger className="h-9 text-xs sm:text-sm bg-background/50 hover:bg-background transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="max-h-[min(50vh,400px)]">
                    {getModelsByProvider(selection.provider).map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center gap-2 max-w-full">
                          <span className="truncate">{model.displayName}</span>

                          {/* Tier Badge */}
                          {model.tier === 'free' ? (
                            <Badge
                              variant="outline"
                              className="text-[10px] px-1.5 h-5 shrink-0 border-green-500/30 text-green-600 dark:text-green-400 bg-green-500/5"
                            >
                              Free
                            </Badge>
                          ) : (
                            <Badge
                              variant="outline"
                              className="text-[10px] px-1.5 h-5 shrink-0 border-amber-500/30 text-amber-600 dark:text-amber-400 bg-amber-500/5"
                            >
                              Paid
                            </Badge>
                          )}

                          {model.isRecommended && (
                            <Badge variant="secondary" className="text-[10px] px-1.5 h-5 shrink-0">
                              Top Pick
                            </Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Model Info */}
            {currentModel && (
              <div className="rounded-md bg-muted/30 p-2 text-xs border border-transparent group-hover:border-border/30 transition-colors">
                <div className="flex flex-wrap items-center justify-between gap-y-1 gap-x-3 text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <span className="flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary/40" />
                      {(currentModel.capabilities.contextWindow / 1000).toFixed(0)}K Context
                    </span>
                    {currentModel.tier && (
                      <span className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500/40" />
                        <span className="capitalize">{currentModel.tier}</span>
                      </span>
                    )}
                  </div>
                  {currentModel.capabilities.supportsVision && (
                    <Badge variant="outline" className="text-[10px] h-5 bg-background/50 px-1.5">
                      Vision
                    </Badge>
                  )}
                </div>
                {currentModel.description && (
                  <p className="mt-1.5 text-xs text-muted-foreground/80 line-clamp-2 leading-relaxed">
                    {currentModel.description}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const tasks: PipelineTask[] = ['intent', 'extraction', 'reasoning', 'summary'];

  if (compact) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className={`font-semibold tracking-tight ${labelClass}`}>Pipeline Models</h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => resetToDefaults()}
            title="Reset to defaults"
            className={`h-6 px-2 hover:bg-destructive/10 hover:text-destructive transition-colors ${textSize === 'sm' ? 'text-[10px]' : 'text-xs'}`}
          >
            <RotateCcw className="h-3 w-3 mr-1" />
            Reset
          </Button>
        </div>
        <div className="space-y-1 border rounded-lg p-1 bg-card/50">{tasks.map(renderTask)}</div>
      </div>
    );
  }

  return (
    <Card className="w-full border-muted/60 shadow-sm hover:shadow-md transition-shadow duration-300">
      <CardHeader className="pb-3 border-b bg-muted/5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0 flex-1">
            <div className="p-2 rounded-lg bg-background border shadow-sm">
              <Settings className="h-5 w-5 text-foreground/70" />
            </div>
            <div className="min-w-0 flex-1">
              <CardTitle className="text-base sm:text-lg tracking-tight">
                Pipeline Configuration
              </CardTitle>
              <CardDescription className="text-xs sm:text-sm mt-1">
                Customize AI models for each stage of the analysis pipeline
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
            <Button
              variant="outline"
              size="sm"
              onClick={() => resetToDefaults()}
              title="Reset all models to defaults"
              className="h-8 text-xs font-medium border-dashed hover:border-solid hover:bg-destructive/5 hover:text-destructive hover:border-destructive/30"
            >
              <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
              Reset Defaults
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-8 w-8 p-0 hover:bg-primary/5"
            >
              <ChevronDown
                className={`h-4 w-4 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
              />
              <span className="sr-only">{isExpanded ? 'Collapse' : 'Expand'}</span>
            </Button>
          </div>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="space-y-4 pt-4 px-3 sm:px-6">
          <div className="grid gap-4">{tasks.map(renderTask)}</div>
        </CardContent>
      )}
    </Card>
  );
}
