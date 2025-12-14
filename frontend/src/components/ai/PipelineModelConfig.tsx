/**
 * Pipeline Model Configuration Panel
 *
 * Allows users to configure which AI model is used for each pipeline task:
 * - Intent Detection
 * - Claim Extraction
 * - Reasoning & Verification
 */

'use client';

import { useState } from 'react';
import { Settings, Zap, FileText, Brain, RotateCcw, ChevronDown, AlignLeft } from 'lucide-react';
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
import { Badge, Separator } from '@/components/ui/primitives';

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
    label: 'Intent & Query Detection',
    description: 'Classify input type and generate search query',
    icon: Zap,
    color: 'text-yellow-500',
  },
  extraction: {
    label: 'Claim Extraction',
    description: 'Extract claims from complex text',
    icon: FileText,
    color: 'text-blue-500',
  },
  summary: {
    label: 'Summarization',
    description: 'Generate executive summary',
    icon: AlignLeft,
    color: 'text-green-500',
  },
  reasoning: {
    label: 'Reasoning & Verification',
    description: 'Complex fact-checking and verification',
    icon: Brain,
    color: 'text-purple-500',
  },
};

export function PipelineModelConfig({ compact = false, textSize = 'md' }: { compact?: boolean; textSize?: 'sm' | 'md' | 'lg' }) {
  const [isExpanded, setIsExpanded] = useState(!compact);
  const { intent, extraction, reasoning, summary, setTaskModel, resetToDefaults } = usePipelineModelsStore();

  const taskSelections = { intent, extraction, reasoning, summary };

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

  const renderTaskConfig = (task: PipelineTask) => {
    const config = taskConfig[task];
    const selection = taskSelections[task];
    const currentModel = getModelById(selection.modelId);
    const isActive = activeTask === task;

    const Icon = config.icon;

    if (compact) {
      return (
        <div key={task} className="py-2 border-b last:border-0">
          <div className="flex items-center gap-2 mb-2">
            <Icon className={`h-4 w-4 ${config.color}`} />
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
                        <span className="text-[10px] text-muted-foreground">★</span>
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
        className={`space-y-3 ${isActive ? 'ring-2 ring-primary/50 rounded-lg p-3 bg-primary/5' : ''}`}
      >
        <div className="flex flex-col sm:flex-row items-start gap-3">
          <Icon className={`mt-1 h-5 w-5 shrink-0 ${config.color}`} />
          <div className="flex-1 w-full space-y-2">
            <div>
              <h4 className="text-sm sm:text-base font-medium">{config.label}</h4>
              <p className="text-xs sm:text-sm text-muted-foreground">{config.description}</p>
            </div>

            {/* Responsive Grid for Provider & Model Selection */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3">
              {/* Provider Selection */}
              <div className="space-y-1">
                <span className="text-xs font-medium text-muted-foreground">Provider</span>
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
                  <SelectTrigger className="h-9 text-xs sm:text-sm">
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
              <div className="space-y-1">
                <span className="text-xs font-medium text-muted-foreground">Model</span>
                <Select
                  value={selection.modelId}
                  onValueChange={(modelId) => setTaskModel(task, selection.provider, modelId)}
                >
                  <SelectTrigger className="h-9 text-xs sm:text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="max-h-[min(50vh,400px)]">
                    {getModelsByProvider(selection.provider).map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center gap-2 max-w-full">
                          <span className="truncate">{model.displayName}</span>
                          {model.isRecommended && (
                            <Badge variant="secondary" className="text-2xs shrink-0">
                              ★
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
              <div className="rounded-md bg-muted/50 p-2 text-xs text-muted-foreground">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-2xs sm:text-xs">
                    Context: {(currentModel.capabilities.contextWindow / 1000).toFixed(0)}K tokens
                  </span>
                  {currentModel.tier && (
                    <Badge variant="outline" className="text-2xs capitalize">
                      {currentModel.tier}
                    </Badge>
                  )}
                  {currentModel.capabilities.supportsVision && (
                    <Badge variant="outline" className="text-2xs">
                      Vision
                    </Badge>
                  )}
                </div>
                {currentModel.description && (
                  <p className="mt-1 text-2xs sm:text-xs line-clamp-2">
                    {currentModel.description}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
        <Separator />
      </div>
    );
  };

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between mb-2">
          <h3 className={`font-medium ${labelClass}`}>Pipeline Models</h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => resetToDefaults()}
            title="Reset to defaults"
            className={`h-6 px-2 ${textSize === 'sm' ? 'text-[10px]' : 'text-xs'}`}
          >
            <RotateCcw className="h-3 w-3 mr-1" />
            Reset
          </Button>
        </div>
        <div className="space-y-1 border rounded-md p-3 bg-card">
          {(Object.keys(taskConfig) as PipelineTask[]).map(renderTaskConfig)}
        </div>
      </div>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-start gap-2 min-w-0 flex-1">
            <Settings className="h-5 w-5 shrink-0 mt-0.5" />
            <div className="min-w-0 flex-1">
              <CardTitle className="text-base sm:text-lg">Pipeline Model Configuration</CardTitle>
              <CardDescription className="text-xs sm:text-sm mt-1">
                Configure AI models for each pipeline task
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => resetToDefaults()}
              title="Reset to defaults"
              className="h-8"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span className="sr-only sm:not-sr-only sm:ml-2">Reset</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-8"
            >
              <ChevronDown
                className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              />
              <span className="sr-only sm:not-sr-only sm:ml-2">
                {isExpanded ? 'Collapse' : 'Expand'}
              </span>
            </Button>
          </div>
        </div>
      </CardHeader>
      {isExpanded && (
        <CardContent className="space-y-4 pt-3">
          {(Object.keys(taskConfig) as PipelineTask[]).map(renderTaskConfig)}
        </CardContent>
      )}
    </Card>
  );
}
