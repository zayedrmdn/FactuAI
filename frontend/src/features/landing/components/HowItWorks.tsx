'use client';

import { motion } from 'framer-motion';
import { FileText, BrainCircuit, Search, CheckCircle2, ArrowRight } from 'lucide-react';

const steps = [
  {
    icon: FileText,
    title: 'Input Your Claim',
    description: 'Paste any statement, news headline, or claim you want verified.',
  },
  {
    icon: BrainCircuit,
    title: 'AI Strategizes',
    description: 'Our LLM generates multi-angle search queries to find relevant evidence.',
  },
  {
    icon: Search,
    title: 'Evidence Gathered',
    description: 'Parallel search across trusted sources with iterative pivot loops.',
  },
  {
    icon: CheckCircle2,
    title: 'Verdict Delivered',
    description: 'Final synthesis with confidence scores and traceable source citations.',
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 bg-muted/20">
      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="mb-16 text-center max-w-2xl mx-auto">
          <motion.span
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-sm font-medium text-primary uppercase tracking-wider"
          >
            How It Works
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mt-3"
          >
            Four steps to truth
          </motion.h2>
        </div>

        {/* Steps Flow - Desktop */}
        <div className="hidden lg:block relative">
          {/* Flowing connection path */}
          <div className="absolute top-1/2 left-0 right-0 -translate-y-1/2 h-px">
            <motion.div
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1, delay: 0.3 }}
              className="h-full bg-gradient-to-r from-border via-primary/40 to-border origin-left"
            />
          </div>

          <div className="relative grid grid-cols-4 gap-6">
            {steps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                className="relative"
              >
                {/* Card with offset for visual depth */}
                <div className="bg-card rounded-2xl border border-border p-6 relative z-10 hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                  {/* Step number badge */}
                  <div className="absolute -top-3 -left-3 w-7 h-7 bg-primary rounded-full flex items-center justify-center text-xs font-bold text-primary-foreground shadow-md">
                    {index + 1}
                  </div>

                  {/* Icon */}
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-5">
                    <step.icon className="w-6 h-6 text-primary" aria-hidden="true" />
                  </div>

                  <h3 className="text-lg font-semibold text-foreground mb-2">{step.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>

                {/* Arrow between cards */}
                {index < steps.length - 1 && (
                  <div className="absolute top-1/2 -right-3 transform -translate-y-1/2 z-20">
                    <motion.div
                      initial={{ opacity: 0, x: -5 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.5 + 0.1 * index }}
                    >
                      <ArrowRight className="w-5 h-5 text-primary" aria-hidden="true" />
                    </motion.div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Steps Flow - Tablet/Mobile */}
        <div className="lg:hidden space-y-4">
          {steps.map((step, index) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.1 * index }}
              className="flex gap-4 items-start"
            >
              {/* Step indicator line */}
              <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-sm font-bold text-primary-foreground">
                  {index + 1}
                </div>
                {index < steps.length - 1 && <div className="w-px h-16 bg-border mt-2" />}
              </div>

              {/* Card */}
              <div className="flex-1 bg-card rounded-xl border border-border p-5 mb-2">
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex-shrink-0 flex items-center justify-center">
                    <step.icon className="w-5 h-5 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-1">{step.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Summary Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="mt-16 bg-card rounded-2xl border border-border p-6 md:p-8"
        >
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
            <div>
              <h3 className="text-xl font-semibold text-foreground mb-2">
                From claim to verdict in seconds
              </h3>
              <p className="text-muted-foreground">
                Our 4-phase pipeline ensures thorough verification without sacrificing speed.
              </p>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <span className="text-3xl font-bold text-primary">4.5s</span>
                <p className="text-xs text-muted-foreground">Avg. Quick Mode</p>
              </div>
              <div className="h-10 w-px bg-border hidden md:block" />
              <div className="text-center">
                <span className="text-3xl font-bold text-info">12s</span>
                <p className="text-xs text-muted-foreground">Avg. Deep Mode</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
