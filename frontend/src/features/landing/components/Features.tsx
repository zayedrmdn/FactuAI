'use client';

import { motion } from 'framer-motion';
import { BrainCircuit, History, Search, Shield, Sparkles, Zap } from 'lucide-react';

const features = [
  {
    icon: BrainCircuit,
    title: 'Multi-Model AI Analysis',
    description:
      'Leverages multiple LLMs with a 4-phase verification pipeline: strategize, search, pivot, verify.',
    size: 'large',
    accent: 'primary',
  },
  {
    icon: Search,
    title: 'Evidence-Based Verdicts',
    description: 'Cross-references claims against fact-check databases and trusted news sources.',
    size: 'medium',
    accent: 'info',
  },
  {
    icon: Zap,
    title: 'Quick & Deep Modes',
    description: 'Fast verification for simple claims, thorough analysis for complex topics.',
    size: 'medium',
    accent: 'warning',
  },
  {
    icon: Shield,
    title: 'Source Filtering',
    description:
      'Automatically blocks unreliable social media sources, prioritizing credible journalism.',
    size: 'small',
    accent: 'success',
  },
  {
    icon: History,
    title: 'Session History',
    description: 'Track and export your verification history.',
    size: 'small',
    accent: 'muted-foreground',
  },
  {
    icon: Sparkles,
    title: 'Confidence Scoring',
    description: 'Transparent confidence levels for every verdict.',
    size: 'small',
    accent: 'primary',
  },
];

const accentClasses: Record<string, string> = {
  primary: 'text-primary bg-primary/10',
  info: 'text-info bg-info/10',
  success: 'text-success bg-success/10',
  warning: 'text-warning bg-warning/10',
  'muted-foreground': 'text-muted-foreground bg-muted',
};

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function Features() {
  const largeFeature = features.find((f) => f.size === 'large')!;
  const mediumFeatures = features.filter((f) => f.size === 'medium');
  const smallFeatures = features.filter((f) => f.size === 'small');

  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-6">
        {/* Section Header - Asymmetric */}
        <div className="mb-16 max-w-2xl">
          <motion.span
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-sm font-medium text-primary uppercase tracking-wider"
          >
            Capabilities
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mt-3 leading-tight"
          >
            Built for precision,
            <br />
            <span className="text-muted-foreground">not speculation.</span>
          </motion.h2>
        </div>

        {/* Bento Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-100px' }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-5"
        >
          {/* Large Feature - Spans 2 columns */}
          <motion.div variants={itemVariants} className="md:col-span-2 lg:row-span-2 group">
            <div className="h-full bg-card rounded-2xl border border-border p-8 lg:p-10 flex flex-col justify-between hover:border-primary/30 transition-colors duration-300">
              <div>
                <div
                  className={`w-14 h-14 rounded-xl flex items-center justify-center ${accentClasses[largeFeature.accent]} mb-6`}
                >
                  <largeFeature.icon className="w-7 h-7" aria-hidden="true" />
                </div>
                <h3 className="text-2xl lg:text-3xl font-semibold text-foreground mb-4">
                  {largeFeature.title}
                </h3>
                <p className="text-muted-foreground text-lg leading-relaxed">
                  {largeFeature.description}
                </p>
              </div>

              {/* Visual element for large card */}
              <div className="mt-8 pt-6 border-t border-border">
                <div className="flex items-center gap-3">
                  <div className="flex -space-x-1">
                    {[1, 2, 3, 4].map((i) => (
                      <div
                        key={i}
                        className="w-8 h-8 rounded-full bg-muted border-2 border-card flex items-center justify-center text-xs font-medium text-muted-foreground"
                      >
                        {i}
                      </div>
                    ))}
                  </div>
                  <span className="text-sm text-muted-foreground">4-Phase Pipeline</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Medium Features */}
          {mediumFeatures.map((feature) => (
            <motion.div key={feature.title} variants={itemVariants} className="group">
              <div className="h-full bg-card rounded-2xl border border-border p-6 lg:p-7 hover:border-primary/30 transition-colors duration-300">
                <div
                  className={`w-11 h-11 rounded-lg flex items-center justify-center ${accentClasses[feature.accent]} mb-5`}
                >
                  <feature.icon className="w-5 h-5" aria-hidden="true" />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </div>
            </motion.div>
          ))}

          {/* Small Features Row */}
          {smallFeatures.map((feature) => (
            <motion.div key={feature.title} variants={itemVariants} className="group">
              <div className="h-full bg-card rounded-2xl border border-border p-5 lg:p-6 hover:border-primary/30 transition-colors duration-300">
                <div className="flex items-start gap-4">
                  <div
                    className={`w-10 h-10 rounded-lg flex-shrink-0 flex items-center justify-center ${accentClasses[feature.accent]}`}
                  >
                    <feature.icon className="w-5 h-5" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-1">{feature.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
