// Full Path: src/features/analyze/components/PipelineStepLoader.tsx
import { useEffect, useState } from 'react';
import { Loader2, Brain, Globe, Search, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

const steps = [
  { id: 'intent', label: 'Analyzing Intent...', icon: Brain, duration: 2000 },
  { id: 'strategy', label: 'Planning Strategy (3 Angles)...', icon: Search, duration: 3000 },
  { id: 'search', label: 'Scanning Knowledge Base...', icon: Globe, duration: 4000 },
  { id: 'verify', label: 'Verifying Claims...', icon: CheckCircle2, duration: 4000 },
];

export function PipelineStepLoader() {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  useEffect(() => {
    if (currentStepIndex >= steps.length - 1) return;
    const step = steps[currentStepIndex];
    if (!step) return;

    const timer = setTimeout(() => {
      setCurrentStepIndex((prev) => prev + 1);
    }, step.duration);

    return () => clearTimeout(timer);
  }, [currentStepIndex]);

  const currentStep = steps[currentStepIndex];
  if (!currentStep) return null;

  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-6 w-full max-w-md mx-auto">
      {/* Icon Animation */}
      <div className="relative">
        <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full animate-pulse" />
        <div className="relative bg-background border rounded-full p-4 shadow-lg">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      </div>

      {/* Steps Visualization */}
      <div className="w-full space-y-4">
        <h3 className="text-center font-medium bg-gradient-to-r from-primary to-info bg-clip-text text-transparent text-lg">
          {currentStep.label}
        </h3>

        <div className="space-y-2">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isActive = idx === currentStepIndex;
            const isCompleted = idx < currentStepIndex;

            return (
              <div
                key={step.id}
                className={`flex items-center gap-3 p-3 rounded-lg border transition-all duration-500 ${
                  isActive
                    ? 'border-primary/50 bg-primary/5 shadow-sm scale-102'
                    : isCompleted
                      ? 'border-transparent opacity-50'
                      : 'border-transparent opacity-30'
                }`}
              >
                <div
                  className={`p-1.5 rounded-full ${
                    isActive || isCompleted
                      ? 'bg-primary/10 text-primary'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {isCompleted ? (
                    <CheckCircle2 className="w-4 h-4" />
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </div>
                <span className={`text-sm font-medium ${isActive ? 'text-primary' : ''}`}>
                  {step.label.replace('...', '')}
                </span>
                {isActive && (
                  <motion.div
                    layoutId="loader-dot"
                    className="ml-auto w-2 h-2 rounded-full bg-primary"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="text-xs text-muted-foreground text-center animate-pulse">
        Reasoning Engine v2.0 Active
      </div>
    </div>
  );
}
