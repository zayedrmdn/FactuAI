'use client';

import { Button } from '@/components/ui/button';
import { BadgeCheck, Lock, Zap } from 'lucide-react';
import Link from 'next/link';

export default function CallToAction() {
  return (
    <section className="py-16 bg-gradient-to-r from-primary to-info">
      <div className="container mx-auto px-6 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <h2 className="text-3xl md:text-5xl font-bold text-primary-foreground mb-6">
            Ready to try FactuAI?
          </h2>
          <p className="text-xl md:text-2xl text-primary-foreground/80 mb-8 leading-relaxed">
            Register for free and start fact-checking instantly. Join researchers, journalists, and
            truth-seekers who trust FactuAI.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link href="/register">
              <Button
                size="lg"
                variant="secondary"
                className="w-full sm:w-auto text-lg px-8 py-4 bg-background text-foreground hover:bg-muted border-0"
              >
                Get Started Free →
              </Button>
            </Link>
            <Link href="/login">
              <button className="w-full sm:w-auto text-lg px-8 py-4 h-12 rounded-md font-medium text-primary-foreground border-2 border-primary-foreground/70 bg-transparent hover:bg-background hover:text-foreground transition-all duration-200 inline-flex items-center justify-center gap-2">
                Already have an account?
              </button>
            </Link>
          </div>

          {/* Feature Highlights */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 text-primary-foreground">
            <div className="flex items-center justify-center gap-3">
              <BadgeCheck className="w-5 h-5" aria-hidden="true" />
              <span className="text-lg">Free to use</span>
            </div>
            <div className="flex items-center justify-center gap-3">
              <Zap className="w-5 h-5" aria-hidden="true" />
              <span className="text-lg">Instant results</span>
            </div>
            <div className="flex items-center justify-center gap-3">
              <Lock className="w-5 h-5" aria-hidden="true" />
              <span className="text-lg">Privacy-focused</span>
            </div>
          </div>

          {/* Stats or Social Proof */}
          <div className="mt-12 pt-8 border-t border-primary-foreground/20">
            <p className="text-primary-foreground/80 text-sm">
              Powered by advanced AI models • Built for accuracy and transparency
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
