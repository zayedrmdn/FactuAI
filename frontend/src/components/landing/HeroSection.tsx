'use client';

import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function HeroSection() {
  return (
    <section className="relative bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 min-h-screen flex items-center">
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Text Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white leading-tight">
                AI-Powered News{' '}
                <span className="text-blue-600 dark:text-blue-400">Summarization</span> &
                Fact-Checking
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 leading-relaxed">
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
            <div className="flex flex-wrap gap-4 text-sm text-gray-500 dark:text-gray-400">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full" aria-hidden="true"></span>
                <span>Built with AI (BERT, T5)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full" aria-hidden="true"></span>
                <span>Research-backed</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-500 rounded-full" aria-hidden="true"></span>
                <span>Privacy-focused</span>
              </div>
            </div>
          </div>

          {/* Right Column - Visual */}
          <div className="relative">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-6 transform rotate-3 hover:rotate-0 transition-transform duration-300">
              {/* Mock UI Preview */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                  <span className="ml-2">FactuAI Analysis</span>
                </div>

                {/* Mock Article */}
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Breaking News Article
                  </h3>
                  <div className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                    <div className="h-2 bg-gray-300 dark:bg-gray-600 rounded w-full"></div>
                    <div className="h-2 bg-gray-300 dark:bg-gray-600 rounded w-3/4"></div>
                    <div className="h-2 bg-gray-300 dark:bg-gray-600 rounded w-1/2"></div>
                  </div>
                </div>

                {/* Mock Summary */}
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-blue-600 dark:text-blue-400">✍️</span>
                    <span className="font-semibold text-blue-900 dark:text-blue-300">
                      AI Summary
                    </span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="h-2 bg-blue-200 dark:bg-blue-700 rounded w-full"></div>
                    <div className="h-2 bg-blue-200 dark:bg-blue-700 rounded w-2/3"></div>
                  </div>
                </div>

                {/* Mock Fact Check */}
                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-green-600 dark:text-green-400">✅</span>
                      <span className="font-semibold text-green-900 dark:text-green-300">
                        Fact Check
                      </span>
                    </div>
                    <span className="text-xs bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-200 px-2 py-1 rounded-full">
                      VERIFIED
                    </span>
                  </div>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    Claim verified with 94% confidence
                  </p>
                </div>
              </div>
            </div>

            {/* Floating Elements */}
            <div className="absolute -top-4 -right-4 bg-blue-500 text-white p-3 rounded-full shadow-lg animate-bounce">
              🔍
            </div>
            <div className="absolute -bottom-4 -left-4 bg-green-500 text-white p-3 rounded-full shadow-lg animate-pulse">
              ✅
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
