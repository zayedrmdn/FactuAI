/**
 * AI Components - Consolidated AI-related components
 * Includes: ActiveModelDisplay, ModelSelector
 */

'use client';

import { useState } from 'react';
import { toast } from 'sonner';
import { Settings, ChevronDown, Check, Sparkles, Zap, FileText, Brain, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuGroup,
} from '@/components/ui/dropdown-menu';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Label, Input, Slider, Textarea } from '@/components/ui/form-controls';
import { Badge } from '@/components/ui/primitives';
import { cn } from '@/lib/utils';
import { useAIStore, useCurrentModelConfig } from '@/stores/ai-store';
import { usePipelineModelsStore } from '@/stores/pipeline-models-store';
import { modelRegistry, getModelById } from '@/config/ai-models';
import type { PipelineTask } from '@/stores/pipeline-models-store';

// ========================================================================================
// ACTIVE MODEL DISPLAY COMPONENT
// ========================================================================================

const taskIcons: Record<PipelineTask, React.ElementType> = {
  intent: Zap,
  extraction: FileText,
  reasoning: Brain,
  summary: FileText,
};

const taskLabels: Record<PipelineTask, string> = {
  intent: 'Intent Detection',
  extraction: 'Claim Extraction',
  reasoning: 'Reasoning',
  summary: 'Summary Generation',
};

export function ActiveModelDisplay() {
  const { activeTask, intent, extraction, reasoning, summary } = usePipelineModelsStore();

  if (!activeTask) return null;

  const taskSelections = { intent, extraction, reasoning, summary };
  const selection = taskSelections[activeTask];
  const model = getModelById(selection.modelId);
  
  // Fallback if model not found (shouldn't happen but prevents crash)
  const modelName = model?.displayName || selection.modelId || 'Unknown Model';

  const TaskIcon = taskIcons[activeTask];

  return (
    <div className="flex items-center gap-2 rounded-lg border bg-card p-3 text-sm">
      <Cpu className="h-4 w-4 text-muted-foreground" />
      <div className="flex items-center gap-2">
        <TaskIcon className="h-4 w-4" />
        <span className="font-medium">{taskLabels[activeTask]}</span>
        <span className="text-muted-foreground">•</span>
        <span className="text-muted-foreground">{modelName}</span>
        <Badge variant="outline" className="text-xs">
          {selection.provider}
        </Badge>
      </div>
    </div>
  );
}

// ========================================================================================
// MODEL SELECTOR COMPONENT
// ========================================================================================

export function ModelSelector() {
  const { selection, setModel, updateOverrides, resetOverrides } = useAIStore();
  const currentModel = useCurrentModelConfig();
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Local state for parameter editing
  const [temperature, setTemperature] = useState(currentModel?.temperature ?? 0.7);
  const [maxTokens, setMaxTokens] = useState(currentModel?.maxTokens ?? 4096);
  const [topP, setTopP] = useState(currentModel?.topP ?? 0.9);
  const [systemPrompt, setSystemPrompt] = useState(currentModel?.systemPrompt ?? '');

  // Sync local state when model changes
  const handleModelChange = (modelId: string) => {
    setModel(modelId);
    const newModel = useAIStore.getState().getCurrentModel();
    if (newModel) {
      setTemperature(newModel.defaultTemperature);
      setMaxTokens(newModel.defaultMaxTokens);
      setTopP(newModel.defaultTopP);
      setSystemPrompt(newModel.defaultSystemPrompt);
      toast.success(`Switched to ${newModel.displayName}`);
    }
  };

  const handleApplySettings = () => {
    updateOverrides({
      temperature,
      maxTokens,
      topP,
      systemPrompt,
    });
    setSettingsOpen(false);
    toast.success('Model settings applied');
  };

  const handleResetSettings = () => {
    resetOverrides();
    if (currentModel) {
      setTemperature(currentModel.defaultTemperature);
      setMaxTokens(currentModel.defaultMaxTokens);
      setTopP(currentModel.defaultTopP);
      setSystemPrompt(currentModel.defaultSystemPrompt);
    }
    setSettingsOpen(false);
    toast.success('Model settings reset to defaults');
  };

  if (!currentModel) {
    return <div className="text-xs text-muted-foreground">No model selected</div>;
  }

  const currentProvider = modelRegistry.providers.find((p) => p.id === selection.provider);
  const hasOverrides = selection.sessionOverrides !== undefined;

  return (
    <div className="flex items-center gap-2 w-full max-w-md">
      {/* Model Selector Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className="h-9 w-full min-w-0 justify-between gap-2 border-border/50 bg-background/50 hover:bg-background"
          >
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <Sparkles className="h-4 w-4 shrink-0 text-primary" />
              <div className="flex flex-col items-start min-w-0 flex-1">
                <span className="text-xs font-medium leading-tight truncate w-full text-left">
                  {currentModel.displayName}
                </span>
                <span className="text-2xs text-muted-foreground leading-tight truncate w-full text-left">
                  {currentProvider?.name}
                </span>
              </div>
            </div>
            <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className="w-[min(calc(100vw-2rem),28rem)] max-h-[70vh] overflow-y-auto"
        >
          {modelRegistry.providers.map((provider) => (
            <DropdownMenuGroup key={provider.id}>
              <DropdownMenuLabel className="flex items-center gap-2">
                <span>{provider.name}</span>
                {provider.id === selection.provider && (
                  <Badge variant="secondary" className="text-2xs px-1.5 py-0">
                    Active
                  </Badge>
                )}
              </DropdownMenuLabel>
              {provider.models.map((model) => (
                <DropdownMenuItem
                  key={model.id}
                  onClick={() => handleModelChange(model.id)}
                  className="flex items-start gap-2 sm:gap-3 py-3 cursor-pointer"
                >
                  <div className="flex h-4 w-4 shrink-0 items-center justify-center mt-1">
                    {model.id === selection.modelId && (
                      <Check className="h-3.5 w-3.5 text-primary" />
                    )}
                  </div>
                  <div className="flex flex-1 flex-col gap-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium">{model.displayName}</span>
                      {model.isRecommended && (
                        <Badge
                          variant="outline"
                          className="text-2xs px-1.5 py-0.5 bg-primary/5 text-primary border-primary/20"
                        >
                          Recommended
                        </Badge>
                      )}
                    </div>
                    {model.description && (
                      <span className="text-xs text-muted-foreground line-clamp-2">
                        {model.description}
                      </span>
                    )}
                    <div className="flex items-center gap-1.5 sm:gap-2 mt-1 flex-wrap">
                      <Badge variant="secondary" className="text-2xs px-1.5 py-0.5">
                        {(model.capabilities.contextWindow / 1000).toFixed(0)}K
                      </Badge>
                      {model.tier && (
                        <Badge
                          variant="outline"
                          className={cn(
                            'text-2xs px-1.5 py-0.5',
                            model.tier === 'free' && 'badge-tier-free',
                            model.tier === 'low' && 'badge-tier-low',
                            model.tier === 'medium' && 'badge-tier-medium',
                            model.tier === 'high' && 'badge-tier-high',
                            model.tier === 'premium' && 'badge-tier-premium'
                          )}
                        >
                          {model.tier}
                        </Badge>
                      )}
                    </div>
                  </div>
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
            </DropdownMenuGroup>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Settings Popover */}
      <Popover open={settingsOpen} onOpenChange={setSettingsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            className={cn(
              'h-9 w-9 shrink-0 border-border/50 bg-background/50 hover:bg-background',
              hasOverrides && 'border-primary/50 bg-primary/5'
            )}
          >
            <Settings className={cn('h-4 w-4', hasOverrides && 'text-primary')} />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-[min(calc(100vw-2rem),25rem)]">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold">Model Parameters</h4>
              {hasOverrides && (
                <Badge variant="secondary" className="text-xs">
                  Custom
                </Badge>
              )}
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="temperature" className="text-xs">
                  Temperature
                </Label>
                <span className="text-xs text-muted-foreground">{temperature.toFixed(2)}</span>
              </div>
              <Slider
                id="temperature"
                min={0}
                max={2}
                step={0.1}
                value={[temperature]}
                onValueChange={([value]) => setTemperature(value ?? 0.7)}
                className="w-full"
              />
              <p className="text-[10px] text-muted-foreground">
                Controls randomness. Lower = more focused, higher = more creative.
              </p>
            </div>

            {/* Max Tokens */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="maxTokens" className="text-xs">
                  Max Tokens
                </Label>
                <Input
                  id="maxTokens"
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="h-7 w-24 text-xs"
                  min={100}
                  max={currentModel.capabilities.contextWindow}
                />
              </div>
              <p className="text-[10px] text-muted-foreground">
                Maximum length of generated response.
              </p>
            </div>

            {/* Top P */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="topP" className="text-xs">
                  Top P
                </Label>
                <span className="text-xs text-muted-foreground">{topP.toFixed(2)}</span>
              </div>
              <Slider
                id="topP"
                min={0}
                max={1}
                step={0.05}
                value={[topP]}
                onValueChange={([value]) => setTopP(value ?? 0.9)}
                className="w-full"
              />
              <p className="text-[10px] text-muted-foreground">
                Nucleus sampling. Lower = more deterministic.
              </p>
            </div>

            {/* System Prompt */}
            <div className="space-y-2">
              <Label htmlFor="systemPrompt" className="text-xs">
                System Prompt
              </Label>
              <Textarea
                id="systemPrompt"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="min-h-[80px] text-xs"
                placeholder="Enter custom system prompt..."
              />
              <p className="text-[10px] text-muted-foreground">
                Define the AI&apos;s role and behavior.
              </p>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" onClick={handleApplySettings} className="flex-1">
                Apply
              </Button>
              <Button size="sm" variant="outline" onClick={handleResetSettings} className="flex-1">
                Reset to Defaults
              </Button>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
