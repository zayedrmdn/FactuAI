'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight, Check, X, AlertTriangle, Loader2 } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { HeroAnimation } from './HeroAnimation';

// Simulated verification animation states
const verificationStates = [
  { text: '"The Earth is approximately 4.5 billion years old"', verdict: 'true', confidence: 94 },
  { text: '"Vaccines cause autism in children"', verdict: 'false', confidence: 98 },
  { text: '"Coffee is the world\'s 2nd most traded commodity"', verdict: 'mixed', confidence: 72 },
];

function VerdictBadge({ verdict }: { verdict: string }) {
  if (verdict === 'true') {
    return (
      <span className="inline-flex items-center gap-1.5 text-success font-semibold">
        <Check className="w-4 h-4" aria-hidden="true" />
        TRUE
      </span>
    );
  }
  if (verdict === 'false') {
    return (
      <span className="inline-flex items-center gap-1.5 text-destructive font-semibold">
        <X className="w-4 h-4" aria-hidden="true" />
        FALSE
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-warning font-semibold">
      <AlertTriangle className="w-4 h-4" aria-hidden="true" />
      MIXED
    </span>
  );
}

export default function HeroSection() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isVerifying, setIsVerifying] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsVerifying(true);
      setTimeout(() => {
        setCurrentIndex((prev) => (prev + 1) % verificationStates.length);
        setIsVerifying(false);
      }, 1200);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const current = verificationStates[currentIndex]!;

  return (
    <section className="relative bg-background min-h-screen flex items-end lg:items-center overflow-hidden">
      {/* Subtle background pattern */}
      <HeroAnimation />

      <div className="container mx-auto px-6 py-24 lg:py-16 relative">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          {/* Left Column - Dramatic Typography (7 cols) */}
          <div className="lg:col-span-7 space-y-8">
            {/* Eyebrow */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/20"
            >
              <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
              <span className="text-sm font-medium text-primary">AI-Powered Fact Verification</span>
            </motion.div>

            {/* Headline - Dramatic Scale */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <h1 className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold text-foreground leading-[0.95] tracking-tight">
                Verify
                <br />
                <span className="text-primary">the truth.</span>
              </h1>
            </motion.div>

            {/* Subheadline */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-lg md:text-xl text-muted-foreground max-w-lg leading-relaxed"
            >
              FactuAI analyzes claims against trusted sources, delivering verdicts backed by
              evidence—not opinions.
            </motion.p>

            {/* CTA Group */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex flex-col sm:flex-row gap-4 pt-4"
            >
              <Link href="/register">
                <Button size="lg" className="w-full sm:w-auto text-base px-8 h-12 gap-2 group">
                  Start Verifying
                  <ArrowRight
                    className="w-4 h-4 group-hover:translate-x-1 transition-transform"
                    aria-hidden="true"
                  />
                </Button>
              </Link>
              <Link href="#how-it-works">
                <Button
                  variant="outline"
                  size="lg"
                  className="w-full sm:w-auto text-base px-8 h-12"
                >
                  See How It Works
                </Button>
              </Link>
            </motion.div>
          </div>

          {/* Right Column - Live Verification Demo (5 cols) */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="lg:col-span-5"
          >
            <div className="relative">
              {/* Main Demo Card */}
              <div className="bg-card rounded-xl border border-border shadow-lg overflow-hidden">
                {/* Header */}
                <div className="px-5 py-4 border-b border-border bg-muted/30">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-destructive/60" />
                    <div className="w-3 h-3 rounded-full bg-warning/60" />
                    <div className="w-3 h-3 rounded-full bg-success/60" />
                    <span className="ml-3 text-sm font-medium text-muted-foreground">
                      Live Verification
                    </span>
                  </div>
                </div>

                {/* Content */}
                <div className="p-6 space-y-5">
                  {/* Claim Display */}
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Claim
                    </span>
                    <p className="text-foreground font-medium leading-relaxed min-h-[3rem]">
                      {current.text}
                    </p>
                  </div>

                  {/* Verdict Display */}
                  <div className="pt-4 border-t border-border">
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                          Verdict
                        </span>
                        <div className="text-xl">
                          {isVerifying ? (
                            <span className="inline-flex items-center gap-2 text-muted-foreground">
                              <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
                              Analyzing...
                            </span>
                          ) : (
                            <VerdictBadge verdict={current.verdict} />
                          )}
                        </div>
                      </div>

                      {!isVerifying && (
                        <motion.div
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.3 }}
                          className="text-right"
                        >
                          <span className="text-xs text-muted-foreground block">Confidence</span>
                          <span className="text-2xl font-bold text-foreground">
                            {current.confidence}%
                          </span>
                        </motion.div>
                      )}
                    </div>
                  </div>

                  {/* Progress Indicator */}
                  <div className="flex gap-1.5 pt-2">
                    {verificationStates.map((_, idx) => (
                      <div
                        key={idx}
                        className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
                          idx === currentIndex ? 'bg-primary' : 'bg-border'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Decorative element - offset card shadow */}
              <div className="absolute -inset-1 -z-10 rounded-xl bg-gradient-to-br from-primary/20 to-transparent blur-xl opacity-40" />
            </div>
          </motion.div>
        </div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 hidden lg:block"
      >
        <div className="w-6 h-10 rounded-full border-2 border-muted-foreground/30 flex items-start justify-center p-2">
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-1 h-2 rounded-full bg-muted-foreground/50"
          />
        </div>
      </motion.div>
    </section>
  );
}
