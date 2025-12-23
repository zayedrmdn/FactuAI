'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Search, Mail, Github, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-card border-t border-border">
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12">
          {/* Brand Column */}
          <div className="md:col-span-5 space-y-6">
            <Link href="/" className="inline-flex items-center gap-3 group">
              <div className="w-9 h-9 bg-primary rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform">
                <Search className="w-4 h-4 text-primary-foreground" aria-hidden="true" />
              </div>
              <span className="text-xl font-bold text-foreground">FactuAI</span>
            </Link>
            <p className="text-muted-foreground leading-relaxed max-w-md">
              AI-powered claim verification built to combat misinformation. Evidence-based verdicts
              you can trust.
            </p>
            <div className="flex items-center gap-4">
              <a
                href="mailto:contact@factual.ai"
                className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center hover:bg-accent transition-colors"
                aria-label="Email"
              >
                <Mail className="w-4 h-4 text-muted-foreground" />
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center hover:bg-accent transition-colors"
                aria-label="GitHub"
              >
                <Github className="w-4 h-4 text-muted-foreground" />
              </a>
            </div>
          </div>

          {/* Links Columns */}
          <div className="md:col-span-3">
            <h3 className="font-semibold text-foreground mb-4">Product</h3>
            <nav className="space-y-3">
              <Link
                href="/register"
                className="block text-muted-foreground hover:text-foreground transition-colors"
              >
                Get Started
              </Link>
              <Link
                href="/login"
                className="block text-muted-foreground hover:text-foreground transition-colors"
              >
                Login
              </Link>
              <Link
                href="#features"
                className="block text-muted-foreground hover:text-foreground transition-colors"
              >
                Features
              </Link>
              <Link
                href="#how-it-works"
                className="block text-muted-foreground hover:text-foreground transition-colors"
              >
                How It Works
              </Link>
            </nav>
          </div>

          <div className="md:col-span-4">
            <h3 className="font-semibold text-foreground mb-4">About</h3>
            <div className="space-y-3 text-muted-foreground">
              <p className="flex items-start gap-2">
                <span className="w-2 h-2 mt-2 rounded-full bg-primary flex-shrink-0" />
                <span>APU Final Year Research Project</span>
              </p>
              <p className="flex items-start gap-2">
                <span className="w-2 h-2 mt-2 rounded-full bg-success flex-shrink-0" />
                <span>SDG Goal 16: Peace, Justice, Strong Institutions</span>
              </p>
              <p className="flex items-start gap-2">
                <span className="w-2 h-2 mt-2 rounded-full bg-info flex-shrink-0" />
                <span>Built by Zayed Ramadhan</span>
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mt-12 pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4"
        >
          <p className="text-sm text-muted-foreground">
            © 2025 FactuAI. Built for truth-seekers everywhere.
          </p>
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <Link href="#" className="hover:text-foreground transition-colors">
              Privacy
            </Link>
            <Link href="#" className="hover:text-foreground transition-colors">
              Terms
            </Link>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
            >
              Source
              <ExternalLink className="w-3 h-3" aria-hidden="true" />
            </a>
          </div>
        </motion.div>
      </div>
    </footer>
  );
}
