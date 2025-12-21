import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  FileText,
  Newspaper,
  Search,
  Sparkles,
} from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      number: '01',
      title: 'Paste or Upload Content',
      description: 'Add text, screenshot, or article snippet through our flexible input system',
      icon: 'file-text',
    },
    {
      number: '02',
      title: 'AI Analysis & Extraction',
      description: 'The system detects main points, extracts claims, and prepares for verification',
      icon: 'brain',
    },
    {
      number: '03',
      title: 'Automated Fact-Check',
      description: 'FactuAI searches databases and classifies claims with confidence scores',
      icon: 'search',
    },
    {
      number: '04',
      title: 'Get Verified Report',
      description: 'Summary, fact verdicts, and evidence sources—all in one comprehensive report',
      icon: 'chart',
    },
  ];

  const getStepIcon = (icon: string) => {
    switch (icon) {
      case 'file-text':
        return <FileText className="w-6 h-6 text-primary" aria-hidden="true" />;
      case 'brain':
        return <BrainCircuit className="w-6 h-6 text-primary" aria-hidden="true" />;
      case 'search':
        return <Search className="w-6 h-6 text-primary" aria-hidden="true" />;
      case 'chart':
        return <BarChart3 className="w-6 h-6 text-primary" aria-hidden="true" />;
      default:
        return <Sparkles className="w-6 h-6 text-primary" aria-hidden="true" />;
    }
  };

  return (
    <section id="how-it-works" className="py-16 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">How It Works</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            From raw content to verified insights in four simple steps
          </p>
        </div>

        {/* Desktop Layout */}
        <div className="hidden lg:block">
          <div className="relative">
            {/* Connection Line */}
            <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-primary via-success to-info transform -translate-y-1/2 z-0"></div>

            <div className="grid grid-cols-4 gap-8 relative z-10">
              {steps.map((step) => (
                <div key={step.number} className="text-center">
                  <div className="bg-card rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center shadow-lg border-4 border-primary">
                    {getStepIcon(step.icon)}
                  </div>
                  <div className="bg-card rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 border border-border">
                    <div className="text-3xl font-bold text-primary mb-2">{step.number}</div>
                    <h3 className="text-xl font-semibold text-foreground mb-3">{step.title}</h3>
                    <p className="text-muted-foreground leading-relaxed">{step.description}</p>
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
              <div className="flex-shrink-0 bg-primary text-primary-foreground rounded-full w-16 h-16 flex items-center justify-center text-xl font-bold">
                {step.number}
              </div>
              <div className="flex-1 bg-card rounded-2xl p-6 shadow-lg border border-border">
                <div className="flex items-center gap-3 mb-3">
                  {getStepIcon(step.icon)}
                  <h3 className="text-xl font-semibold text-foreground">{step.title}</h3>
                </div>
                <p className="text-muted-foreground leading-relaxed">{step.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Demo Preview */}
        <div className="mt-16 bg-card rounded-2xl p-8 shadow-xl border border-border">
          <h3 className="text-2xl font-bold text-foreground text-center mb-8">See It In Action</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div className="space-y-4">
              <div className="bg-muted/30 rounded-lg p-4 h-32 flex items-center justify-center border border-border">
                <Newspaper className="w-10 h-10 text-muted-foreground" aria-hidden="true" />
              </div>
              <p className="text-sm text-muted-foreground">Raw Article</p>
            </div>

            <div className="flex items-center justify-center">
              <ArrowRight className="w-8 h-8 text-primary animate-pulse" aria-hidden="true" />
            </div>

            <div className="space-y-4">
              <div className="bg-success/10 rounded-lg p-4 h-32 flex items-center justify-center border border-success/20">
                <CheckCircle2 className="w-10 h-10 text-success" aria-hidden="true" />
              </div>
              <p className="text-sm text-muted-foreground">Verified Report</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
