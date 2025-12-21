import {
  BrainCircuit,
  History,
  Search,
  Upload,
  GraduationCap,
  Settings2,
  ScrollText,
} from 'lucide-react';

export default function Features() {
  const features = [
    {
      icon: 'brain',
      title: 'Smart Summarization',
      description:
        'Extractive + abstractive AI models for context-rich news briefs and key insights.',
    },
    {
      icon: 'search',
      title: 'Instant Fact-Checking',
      description: 'Get automated True/False/Unverified verdicts on claims with confidence scores.',
    },
    {
      icon: 'upload',
      title: 'Flexible Input',
      description: 'Paste text, upload documents, or analyze screenshots with our OCR technology.',
    },
    {
      icon: 'history',
      title: 'Session History',
      description: "Track what you've checked. Export reports to PDF anytime for reference.",
    },
  ];

  const getFeatureIcon = (icon: string) => {
    switch (icon) {
      case 'brain':
        return <BrainCircuit className="w-10 h-10 text-primary" aria-hidden="true" />;
      case 'search':
        return <Search className="w-10 h-10 text-info" aria-hidden="true" />;
      case 'upload':
        return <Upload className="w-10 h-10 text-success" aria-hidden="true" />;
      case 'history':
        return <History className="w-10 h-10 text-warning" aria-hidden="true" />;
      default:
        return <Search className="w-10 h-10 text-primary" aria-hidden="true" />;
    }
  };

  return (
    <section className="py-16 bg-background">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            What FactuAI Can Do for You
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Powerful AI tools designed to help you navigate information with confidence
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group bg-card rounded-2xl p-6 text-center hover:shadow-xl hover:scale-105 transition-all duration-300 border border-border"
            >
              <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                {getFeatureIcon(feature.icon)}
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Additional Feature Highlight */}
        <div className="mt-16 bg-muted/30 rounded-2xl p-8 text-center border border-border">
          <h3 className="text-2xl font-bold text-foreground mb-4">
            Built for Truth-Seekers, Researchers, and Everyday Readers
          </h3>
          <p className="text-muted-foreground max-w-3xl mx-auto mb-6">
            FactuAI combines cutting-edge AI research with practical tools to help you make informed
            decisions based on verified information.
          </p>
          <div className="flex flex-wrap justify-center gap-6 text-sm">
            <div className="flex items-center gap-2 bg-background px-4 py-2 rounded-full shadow-sm border border-border">
              <Settings2 className="w-4 h-4 text-primary" aria-hidden="true" />
              <span className="text-muted-foreground">Built with AI (BERT, T5, QLoRA)</span>
            </div>
            <div className="flex items-center gap-2 bg-background px-4 py-2 rounded-full shadow-sm border border-border">
              <GraduationCap className="w-4 h-4 text-success" aria-hidden="true" />
              <span className="text-muted-foreground">Research-backed Development</span>
            </div>
            <div className="flex items-center gap-2 bg-background px-4 py-2 rounded-full shadow-sm border border-border">
              <ScrollText className="w-4 h-4 text-info" aria-hidden="true" />
              <span className="text-muted-foreground">SDG Goal 16 Compliant</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
