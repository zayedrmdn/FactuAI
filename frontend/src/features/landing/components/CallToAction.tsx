'use client';

import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { ArrowRight, Quote } from 'lucide-react';
import Link from 'next/link';

export default function CallToAction() {
  return (
    <section className="py-24 bg-background relative overflow-hidden">
      {/* Subtle accent background */}
      <div className="absolute inset-0 opacity-[0.03]">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-primary to-transparent" />
      </div>

      <div className="container mx-auto px-6 relative">
        <div className="max-w-4xl mx-auto">
          {/* Editorial Quote Style */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center space-y-8"
          >
            {/* Quote Mark */}
            <div className="flex justify-center">
              <Quote className="w-12 h-12 text-primary/30 rotate-180" aria-hidden="true" />
            </div>

            {/* Main Quote / Statement */}
            <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-foreground leading-relaxed">
              In an era of misinformation, having a reliable fact-checking tool is not a luxury—
              <span className="text-primary"> it is a necessity.</span>
            </h2>

            {/* Divider */}
            <div className="flex items-center justify-center gap-4">
              <div className="h-px w-16 bg-border" />
              <span className="text-sm text-muted-foreground font-medium">FactuAI</span>
              <div className="h-px w-16 bg-border" />
            </div>

            {/* CTA */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="pt-4"
            >
              <Link href="/register">
                <Button size="lg" className="text-base px-10 h-13 gap-2 group">
                  Create Free Account
                  <ArrowRight
                    className="w-4 h-4 group-hover:translate-x-1 transition-transform"
                    aria-hidden="true"
                  />
                </Button>
              </Link>
              <p className="mt-4 text-sm text-muted-foreground">
                No credit card required • Start verifying in seconds
              </p>
            </motion.div>
          </motion.div>

          {/* Trust Indicators */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="mt-16 pt-8 border-t border-border"
          >
            <div className="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-12 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-success" />
                <span>Open-source powered</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-info" />
                <span>Multi-model verification</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary" />
                <span>Research-backed methodology</span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
