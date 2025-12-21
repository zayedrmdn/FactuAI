'use client';

import { Button } from '@/components/ui/button';
import { CheckCircle2, FilePenLine, Search } from 'lucide-react';
import Link from 'next/link';

export default function HeroSection() {
  return (
    <section className="relative bg-gradient-to-br from-background to-muted/40 min-h-screen flex items-center">
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Text Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <h1 className="text-4xl md:text-6xl font-bold text-foreground leading-tight">
                AI-Powered News <span className="text-primary">Summarization</span> & Fact-Checking
              </h1>
              <p className="text-xl text-muted-foreground leading-relaxed">
                FactuAI helps you cut through misinformation by providing concise summaries and
                verified facts backed by real evidence.
              </p>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Link href="/register">
                <Button size="lg" className="w-full sm:w-auto text-lg px-8 py-3">
                  Try It Now →
                </Button>
              </Link>
              <Link href="#how-it-works">
                <Button variant="outline" size="lg" className="w-full sm:w-auto text-lg px-8 py-3">
                  Learn How It Works
                </Button>
              </Link>
            </div>

            {/* Trust Indicators */}
            <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-success rounded-full" aria-hidden="true"></span>
                <span>Built with AI (BERT, T5)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-info rounded-full" aria-hidden="true"></span>
                <span>Research-backed</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-primary rounded-full" aria-hidden="true"></span>
                <span>Privacy-focused</span>
              </div>
            </div>
          </div>

          {/* Right Column - Visual */}
          <div className="relative">
            <div className="bg-card rounded-2xl shadow-2xl p-6 transform rotate-3 hover:rotate-0 transition-transform duration-300 border border-border">
              {/* Mock UI Preview */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <div className="w-3 h-3 bg-destructive/60 rounded-full" aria-hidden="true"></div>
                  <div className="w-3 h-3 bg-warning/60 rounded-full" aria-hidden="true"></div>
                  <div className="w-3 h-3 bg-success/60 rounded-full" aria-hidden="true"></div>
                  <span className="ml-2">FactuAI Analysis</span>
                </div>

                {/* Mock Article */}
                <div className="bg-muted/30 p-4 rounded-lg border border-border">
                  <h3 className="font-semibold text-foreground mb-2">Breaking News Article</h3>
                  <div className="space-y-2 text-sm text-muted-foreground">
                    <div className="h-2 bg-muted rounded w-full"></div>
                    <div className="h-2 bg-muted rounded w-3/4"></div>
                    <div className="h-2 bg-muted rounded w-1/2"></div>
                  </div>
                </div>

                {/* Mock Summary */}
                <div className="bg-info/10 p-4 rounded-lg border border-info/20">
                  <div className="flex items-center gap-2 mb-2">
                    <FilePenLine className="w-4 h-4 text-info" aria-hidden="true" />
                    <span className="font-semibold text-foreground">AI Summary</span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="h-2 bg-info/20 rounded w-full"></div>
                    <div className="h-2 bg-info/20 rounded w-2/3"></div>
                  </div>
                </div>

                {/* Mock Fact Check */}
                <div className="bg-success/10 p-4 rounded-lg border border-success/20">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-success" aria-hidden="true" />
                      <span className="font-semibold text-foreground">Fact Check</span>
                    </div>
                    <span className="text-xs bg-success/15 text-success px-2 py-1 rounded-full">
                      VERIFIED
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Claim verified with 94% confidence
                  </p>
                </div>
              </div>
            </div>

            {/* Floating Elements */}
            <div className="absolute -top-4 -right-4 bg-info text-info-foreground p-3 rounded-full shadow-lg animate-bounce">
              <Search className="w-5 h-5" aria-hidden="true" />
            </div>
            <div className="absolute -bottom-4 -left-4 bg-success text-success-foreground p-3 rounded-full shadow-lg animate-pulse">
              <CheckCircle2 className="w-5 h-5" aria-hidden="true" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
