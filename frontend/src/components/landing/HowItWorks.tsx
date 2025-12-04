export default function HowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Paste or Upload Content",
      description: "Add text, screenshot, or article snippet through our flexible input system",
      icon: "📝"
    },
    {
      number: "02", 
      title: "AI Analysis & Extraction",
      description: "The system detects main points, extracts claims, and prepares for verification",
      icon: "🧠"
    },
    {
      number: "03",
      title: "Automated Fact-Check",
      description: "FactuAI searches databases and classifies claims with confidence scores",
      icon: "🔍"
    },
    {
      number: "04",
      title: "Get Verified Report",
      description: "Summary, fact verdicts, and evidence sources—all in one comprehensive report",
      icon: "📊"
    }
  ];

  return (
    <section id="how-it-works" className="py-16 bg-gray-50 dark:bg-gray-800">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
            How It Works
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            From raw content to verified insights in four simple steps
          </p>
        </div>

        {/* Desktop Layout */}
        <div className="hidden lg:block">
          <div className="relative">
            {/* Connection Line */}
            <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 via-green-500 to-purple-500 transform -translate-y-1/2 z-0"></div>
            
            <div className="grid grid-cols-4 gap-8 relative z-10">
              {steps.map((step) => (
                <div key={step.number} className="text-center">
                  <div className="bg-white dark:bg-gray-900 rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center shadow-lg border-4 border-blue-500 dark:border-blue-400">
                    <span className="text-2xl">{step.icon}</span>
                  </div>
                  <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
                    <div className="text-3xl font-bold text-blue-500 dark:text-blue-400 mb-2">
                      {step.number}
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                      {step.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Mobile/Tablet Layout */}
        <div className="lg:hidden space-y-8">
          {steps.map((step) => (
            <div key={step.number} className="flex items-start gap-6">
              <div className="flex-shrink-0 bg-blue-500 text-white rounded-full w-16 h-16 flex items-center justify-center text-xl font-bold">
                {step.number}
              </div>
              <div className="flex-1 bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-3">
                  <span className="text-2xl">{step.icon}</span>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                    {step.title}
                  </h3>
                </div>
                <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Demo Preview */}
        <div className="mt-16 bg-white dark:bg-gray-900 rounded-2xl p-8 shadow-xl">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white text-center mb-8">
            See It In Action
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div className="space-y-4">
              <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 h-32 flex items-center justify-center">
                <span className="text-4xl">📰</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">Raw Article</p>
            </div>
            
            <div className="flex items-center justify-center">
              <div className="text-3xl text-blue-500 animate-pulse">→</div>
            </div>
            
            <div className="space-y-4">
              <div className="bg-green-100 dark:bg-green-900/20 rounded-lg p-4 h-32 flex items-center justify-center">
                <span className="text-4xl">✅</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">Verified Report</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
