export default function Features() {
  const features = [
    {
      icon: '🧠',
      title: 'Smart Summarization',
      description:
        'Extractive + abstractive AI models for context-rich news briefs and key insights.',
    },
    {
      icon: '🔍',
      title: 'Instant Fact-Checking',
      description: 'Get automated True/False/Unverified verdicts on claims with confidence scores.',
    },
    {
      icon: '📂',
      title: 'Flexible Input',
      description: 'Paste text, upload documents, or analyze screenshots with our OCR technology.',
    },
    {
      icon: '🕘',
      title: 'Session History',
      description: "Track what you've checked. Export reports to PDF anytime for reference.",
    },
  ];

  return (
    <section className="py-16 bg-white dark:bg-gray-900">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
            What FactuAI Can Do for You
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Powerful AI tools designed to help you navigate information with confidence
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group bg-gray-50 dark:bg-gray-800 rounded-2xl p-6 text-center hover:shadow-xl hover:scale-105 transition-all duration-300 border border-gray-100 dark:border-gray-700"
            >
              <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Additional Feature Highlight */}
        <div className="mt-16 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-8 text-center border border-blue-100 dark:border-blue-800">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Built for Truth-Seekers, Researchers, and Everyday Readers
          </h3>
          <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-6">
            FactuAI combines cutting-edge AI research with practical tools to help you make informed
            decisions based on verified information.
          </p>
          <div className="flex flex-wrap justify-center gap-6 text-sm">
            <div className="flex items-center gap-2 bg-white dark:bg-gray-800 px-4 py-2 rounded-full shadow-sm">
              <span className="text-blue-500">⚙️</span>
              <span className="text-gray-700 dark:text-gray-300">
                Built with AI (BERT, T5, QLoRA)
              </span>
            </div>
            <div className="flex items-center gap-2 bg-white dark:bg-gray-800 px-4 py-2 rounded-full shadow-sm">
              <span className="text-green-500">🎓</span>
              <span className="text-gray-700 dark:text-gray-300">Research-backed Development</span>
            </div>
            <div className="flex items-center gap-2 bg-white dark:bg-gray-800 px-4 py-2 rounded-full shadow-sm">
              <span className="text-purple-500">📜</span>
              <span className="text-gray-700 dark:text-gray-300">SDG Goal 16 Compliant</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
