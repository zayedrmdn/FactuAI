import Link from "next/link";

export default function Footer() {
  return (
    <footer className="bg-gray-900 dark:bg-black text-white py-12">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Left - Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                <span className="font-bold text-white">F</span>
              </div>
              <span className="text-xl font-bold">FactuAI</span>
            </div>
            <p className="text-gray-400 leading-relaxed">
              AI-powered news summarization and fact-checking platform. 
              Helping you navigate information with confidence and clarity.
            </p>
            <div className="flex items-center gap-4 text-sm text-gray-400">
              <span>🎓 Research Project</span>
              <span>🔬 AI-Powered</span>
              <span>🌍 Global Impact</span>
            </div>
          </div>

          {/* Center - Quick Links */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Quick Links</h3>
            <nav className="space-y-2">
              <Link href="/" className="block text-gray-400 hover:text-white transition-colors">
                Home
              </Link>
              <Link href="/register" className="block text-gray-400 hover:text-white transition-colors">
                Get Started
              </Link>
              <Link href="/login" className="block text-gray-400 hover:text-white transition-colors">
                Login
              </Link>
              <Link href="#how-it-works" className="block text-gray-400 hover:text-white transition-colors">
                How It Works
              </Link>
            </nav>
          </div>

          {/* Right - Contact & Info */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Contact & Info</h3>
            <div className="space-y-2 text-gray-400">
              <p className="flex items-center gap-2">
                <span>🎓</span>
                APU Final Year Project
              </p>
              <p className="flex items-center gap-2">
                <span>🏗️</span>
                Built by Zayed Ramadhan
              </p>
              <p className="flex items-center gap-2">
                <span>📧</span>
                Contact for inquiries
              </p>
              <p className="flex items-center gap-2">
                <span>📜</span>
                SDG Goal 16 - Peace & Justice
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-gray-800">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-gray-400 text-sm">
              © 2025 FactuAI. Built with ❤️ for truth-seekers everywhere.
            </p>
            <div className="flex items-center gap-6 text-sm text-gray-400">
              <Link href="#" className="hover:text-white transition-colors">
                Privacy Policy
              </Link>
              <Link href="#" className="hover:text-white transition-colors">
                Terms of Service
              </Link>
              <Link href="#" className="hover:text-white transition-colors">
                Help
              </Link>
            </div>
          </div>
        </div>

        {/* Tech Stack Badge */}
        <div className="mt-8 text-center">
          <div className="inline-flex items-center gap-2 bg-gray-800 px-4 py-2 rounded-full text-xs text-gray-400">
            <span>⚡ Built with</span>
            <span className="text-blue-400">Next.js</span>
            <span>•</span>
            <span className="text-green-400">Flask</span>
            <span>•</span>
            <span className="text-purple-400">AI Models</span>
            <span>•</span>
            <span className="text-yellow-400">PostgreSQL</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
