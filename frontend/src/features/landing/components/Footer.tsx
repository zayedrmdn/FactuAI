import Link from 'next/link';
import { GraduationCap, Heart, Mail, Microscope, Globe2, Hammer, ScrollText } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-muted/30 text-foreground py-12 border-t border-border">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Left - Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="font-bold text-primary-foreground">F</span>
              </div>
              <span className="text-xl font-bold">FactuAI</span>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              AI-powered news summarization and fact-checking platform. Helping you navigate
              information with confidence and clarity.
            </p>
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <GraduationCap className="w-4 h-4" aria-hidden="true" />
                Research Project
              </span>
              <span className="inline-flex items-center gap-2">
                <Microscope className="w-4 h-4" aria-hidden="true" />
                AI-Powered
              </span>
              <span className="inline-flex items-center gap-2">
                <Globe2 className="w-4 h-4" aria-hidden="true" />
                Global Impact
              </span>
            </div>
          </div>

          {/* Center - Quick Links */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Quick Links</h3>
            <nav className="space-y-2">
              <Link
                href="/"
                className="block text-muted-foreground hover:text-foreground transition-colors"
              >
                Home
              </Link>
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
                href="#how-it-works"
                className="block text-muted-foreground hover:text-foreground transition-colors"
              >
                How It Works
              </Link>
            </nav>
          </div>

          {/* Right - Contact & Info */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Contact & Info</h3>
            <div className="space-y-2 text-muted-foreground">
              <p className="flex items-center gap-2">
                <GraduationCap className="w-4 h-4" aria-hidden="true" />
                <span>APU Final Year Project</span>
              </p>
              <p className="flex items-center gap-2">
                <Hammer className="w-4 h-4" aria-hidden="true" />
                <span>Built by Zayed Ramadhan</span>
              </p>
              <p className="flex items-center gap-2">
                <Mail className="w-4 h-4" aria-hidden="true" />
                <span>Contact for inquiries</span>
              </p>
              <p className="flex items-center gap-2">
                <ScrollText className="w-4 h-4" aria-hidden="true" />
                <span>SDG Goal 16 - Peace & Justice</span>
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-border">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-muted-foreground text-sm inline-flex items-center gap-2">
              <span>© 2025 FactuAI. Built with</span>
              <Heart className="w-4 h-4" aria-hidden="true" />
              <span>for truth-seekers everywhere.</span>
            </p>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <Link href="#" className="hover:text-foreground transition-colors">
                Privacy Policy
              </Link>
              <Link href="#" className="hover:text-foreground transition-colors">
                Terms of Service
              </Link>
              <Link href="#" className="hover:text-foreground transition-colors">
                Help
              </Link>
            </div>
          </div>
        </div>

        {/* Tech Stack Badge */}
        <div className="mt-8 text-center">
          <div className="inline-flex flex-wrap items-center gap-2 bg-card px-4 py-2 rounded-full text-xs text-muted-foreground border border-border">
            <span>Built with</span>
            <span className="text-primary">Next.js</span>
            <span>•</span>
            <span className="text-info">Flask</span>
            <span>•</span>
            <span className="text-success">AI Models</span>
            <span>•</span>
            <span className="text-warning">PostgreSQL</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
