// Full Path: src/features/analyze/components/PipelineStepLoader.tsx
'use client';

import { useEffect, useState, useMemo } from 'react';
import { Brain, Globe, Search, Shield, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const steps = [
  {
    id: 'intent',
    label: 'Extracting Claims',
    sublabel: 'Understanding what you want verified',
    icon: Brain,
    duration: 2000,
    color: 'primary',
  },
  {
    id: 'strategy',
    label: 'Strategizing',
    sublabel: 'Planning multi-angle search queries',
    icon: Search,
    duration: 3000,
    color: 'info',
  },
  {
    id: 'search',
    label: 'Gathering Evidence',
    sublabel: 'Scanning trusted knowledge sources',
    icon: Globe,
    duration: 4000,
    color: 'success',
  },
  {
    id: 'verify',
    label: 'Synthesizing Verdict',
    sublabel: 'Cross-referencing and scoring',
    icon: Shield,
    duration: 4000,
    color: 'warning',
  },
];

// Orbital particle component
function OrbitalParticle({
  delay,
  size,
  duration,
  radius,
}: {
  delay: number;
  size: number;
  duration: number;
  radius: number;
}) {
  return (
    <motion.div
      className="absolute rounded-full bg-primary/40"
      style={{
        width: size,
        height: size,
        top: '50%',
        left: '50%',
        marginTop: -size / 2,
        marginLeft: -size / 2,
      }}
      animate={{
        x: [
          Math.cos(0) * radius,
          Math.cos(Math.PI / 2) * radius,
          Math.cos(Math.PI) * radius,
          Math.cos((Math.PI * 3) / 2) * radius,
          Math.cos(Math.PI * 2) * radius,
        ],
        y: [
          Math.sin(0) * radius,
          Math.sin(Math.PI / 2) * radius,
          Math.sin(Math.PI) * radius,
          Math.sin((Math.PI * 3) / 2) * radius,
          Math.sin(Math.PI * 2) * radius,
        ],
        opacity: [0.3, 0.8, 0.3, 0.8, 0.3],
      }}
      transition={{
        duration,
        repeat: Infinity,
        delay,
        ease: 'linear',
      }}
    />
  );
}

export function PipelineStepLoader() {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [progress, setProgress] = useState(0);

  // Progress animation within each step
  useEffect(() => {
    const step = steps[currentStepIndex];
    if (!step) return;

    const interval = setInterval(() => {
      setProgress((prev) => {
        const increment = 100 / (step.duration / 50);
        return Math.min(prev + increment, 100);
      });
    }, 50);

    return () => clearInterval(interval);
  }, [currentStepIndex]);

  // Step progression
  useEffect(() => {
    if (currentStepIndex >= steps.length - 1) return;
    const step = steps[currentStepIndex];
    if (!step) return;

    const timer = setTimeout(() => {
      setCurrentStepIndex((prev) => prev + 1);
      setProgress(0);
    }, step.duration);

    return () => clearTimeout(timer);
  }, [currentStepIndex]);

  const currentStep = steps[currentStepIndex];
  const CurrentIcon = currentStep?.icon ?? Sparkles;

  // Generate random particles
  const particles = useMemo(
    () =>
      Array.from({ length: 6 }).map((_, i) => ({
        delay: i * 0.5,
        size: 4 + Math.random() * 4,
        duration: 3 + Math.random() * 2,
        radius: 50 + Math.random() * 20,
      })),
    []
  );

  if (!currentStep) return null;

  const totalProgress = (currentStepIndex / steps.length) * 100 + progress / steps.length;

  return (
    <div className="flex flex-col items-center justify-center p-8 w-full max-w-lg mx-auto">
      {/* Central Orb with Orbital Animation */}
      <div className="relative w-40 h-40 mb-8">
        {/* Outer glow ring */}
        <motion.div
          className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/20 to-info/20 blur-2xl"
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />

        {/* Orbiting particles */}
        {particles.map((particle, i) => (
          <OrbitalParticle key={i} {...particle} />
        ))}

        {/* Progress ring */}
        <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 100 100">
          {/* Background track */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-border"
          />
          {/* Progress arc */}
          <motion.circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="url(#progressGradient)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * 45}`}
            animate={{
              strokeDashoffset: 2 * Math.PI * 45 * (1 - totalProgress / 100),
            }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
          />
          <defs>
            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="var(--primary)" />
              <stop offset="100%" stopColor="var(--info)" />
            </linearGradient>
          </defs>
        </svg>

        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep.id}
              initial={{ scale: 0.5, opacity: 0, rotate: -180 }}
              animate={{ scale: 1, opacity: 1, rotate: 0 }}
              exit={{ scale: 0.5, opacity: 0, rotate: 180 }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              className="w-16 h-16 rounded-2xl bg-card border border-border shadow-lg flex items-center justify-center"
            >
              <CurrentIcon className="w-8 h-8 text-primary" />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Current Step Label */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
          className="text-center mb-8"
        >
          <h3 className="text-2xl font-bold text-foreground mb-1">{currentStep.label}</h3>
          <p className="text-sm text-muted-foreground">{currentStep.sublabel}</p>
        </motion.div>
      </AnimatePresence>

      {/* Step Progress Dots */}
      <div className="flex items-center gap-3 mb-6">
        {steps.map((step, idx) => {
          const isActive = idx === currentStepIndex;
          const isCompleted = idx < currentStepIndex;

          return (
            <div key={step.id} className="flex items-center">
              <motion.div
                className={`relative flex items-center justify-center`}
                animate={{
                  scale: isActive ? 1 : 0.8,
                }}
                transition={{ duration: 0.2 }}
              >
                {/* Background circle */}
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center transition-colors duration-300 ${
                    isCompleted
                      ? 'bg-primary text-primary-foreground'
                      : isActive
                        ? 'bg-primary/20 text-primary border-2 border-primary'
                        : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {isCompleted ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    >
                      <Sparkles className="w-4 h-4" />
                    </motion.div>
                  ) : (
                    <span className="text-xs font-bold">{idx + 1}</span>
                  )}
                </div>

                {/* Active indicator pulse */}
                {isActive && (
                  <motion.div
                    className="absolute inset-0 rounded-full border-2 border-primary"
                    animate={{
                      scale: [1, 1.3, 1],
                      opacity: [0.5, 0, 0.5],
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      ease: 'easeOut',
                    }}
                  />
                )}
              </motion.div>

              {/* Connector line */}
              {idx < steps.length - 1 && (
                <div className="w-6 h-0.5 mx-1">
                  <motion.div
                    className="h-full bg-primary rounded-full origin-left"
                    initial={{ scaleX: 0 }}
                    animate={{
                      scaleX: isCompleted ? 1 : isActive ? progress / 100 : 0,
                    }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Overall Progress */}
      <div className="w-full max-w-xs">
        <div className="flex justify-between text-xs text-muted-foreground mb-2">
          <span>
            Step {currentStepIndex + 1} of {steps.length}
          </span>
          <span>{Math.round(totalProgress)}%</span>
        </div>
        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-primary to-info rounded-full"
            animate={{ width: `${totalProgress}%` }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Subtle branding */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="mt-6 text-xs text-muted-foreground/60"
      >
        FactuAI 4-Phase Pipeline
      </motion.p>
    </div>
  );
}
