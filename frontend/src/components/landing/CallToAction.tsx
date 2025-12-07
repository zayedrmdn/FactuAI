'use client';

import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function CallToAction() {
  return (
    <section className="py-16 bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-800 dark:to-indigo-900">
      <div className="container mx-auto px-6 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Ready to try FactuAI?</h2>
          <p className="text-xl md:text-2xl text-blue-100 mb-8 leading-relaxed">
            Register for free and start fact-checking instantly. Join researchers, journalists, and
            truth-seekers who trust FactuAI.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link href="/register">
              <Button
                size="lg"
                variant="secondary"
                className="w-full sm:w-auto text-lg px-8 py-4 bg-white text-blue-600 hover:bg-blue-50 border-0"
              >
                Get Started Free →
              </Button>
            </Link>
            <Link href="/login">
              <button className="w-full sm:w-auto text-lg px-8 py-4 h-12 rounded-md font-medium text-white border-2 border-white bg-transparent hover:bg-white hover:text-blue-600 transition-all duration-200 inline-flex items-center justify-center gap-2">
                Already have an account?
              </button>
            </Link>
          </div>

          {/* Feature Highlights */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 text-white">
            <div className="flex items-center justify-center gap-3">
              <span className="text-2xl">🆓</span>
              <span className="text-lg">Free to use</span>
            </div>
            <div className="flex items-center justify-center gap-3">
              <span className="text-2xl">⚡</span>
              <span className="text-lg">Instant results</span>
            </div>
            <div className="flex items-center justify-center gap-3">
              <span className="text-2xl">🔒</span>
              <span className="text-lg">Privacy-focused</span>
            </div>
          </div>

          {/* Stats or Social Proof */}
          <div className="mt-12 pt-8 border-t border-blue-500/30">
            <p className="text-blue-100 text-sm">
              Powered by advanced AI models • Built for accuracy and transparency
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
