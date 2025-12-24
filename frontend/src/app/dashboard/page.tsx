// Path: frontend/src/app/dashboard/page.tsx
'use client';

import { useState } from 'react';
import { SettingsDialog } from '@/features/dashboard-shell';
import { InputCard, ResultsView } from '@/features/analyze';
import { useSettings } from '@/lib/hooks/useSettings';
import { useAppState } from '@/lib/hooks/useAppState';
import { useHistory } from '@/features/history';
import { cn } from '@/lib/utils';
import { DashboardHero } from './components/DashboardHero';
import { RecentActivityCard } from './components/RecentActivityCard';
import { Shield, Scan, FileCheck } from 'lucide-react';

export default function DashboardPage() {
  const { prefs, savePrefs, isDark, toggleTheme, isLoaded: settingsLoaded } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { history } = useHistory();

  const {
    input,
    setInput,
    showResults,
    factResults,
    summary,
    updated,
    avgConfidence,
    loading,
    loadingPhase,
    progress,
    currentClaim,
    factCheckError,
    handleFactCheck,
    handleCancel,
    handleRetryInput,
    handleClear,
    handleAIDetection,
    handleInputTypeChange,
    saveImageToHistory,
    saveVideoToHistory,
  } = useAppState();

  // Quick action chips - distinctive categories
  const quickActions = [
    { label: 'Political Claim', icon: Shield, color: 'text-chart-4' },
    { label: 'Deepfake Scan', icon: Scan, color: 'text-destructive' },
    { label: 'Source Check', icon: FileCheck, color: 'text-warning' },
  ];

  const handleQuickAction = (action: string) => {
    const samples: Record<string, string> = {
      'Political Claim':
        'The government announced a 50% reduction in taxes for all citizens starting next year.',
      'Deepfake Scan':
        'This video shows a politician making controversial statements about climate change.',
      'Source Check':
        'According to a study by Harvard University, coffee can extend lifespan by 10 years.',
    };
    setInput(samples[action] || '');
  };

  if (!settingsLoaded) {
    return (
      <div className="min-h-full flex items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <>
      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        prefs={prefs}
        savePrefs={savePrefs}
        toggleTheme={toggleTheme}
        isDark={isDark}
      />

      {/* Main Container - Scrollable */}
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-12 flex flex-col gap-10">
          {/* Show results when available, otherwise show composer */}
          {showResults ? (
            <ResultsView
              results={factResults}
              summary={summary}
              updated={updated}
              loading={loading}
              loadingPhase={loadingPhase}
              progress={progress}
              currentClaim={currentClaim}
              prefs={prefs}
              averageConfidence={avgConfidence}
              onRetry={handleRetryInput}
              onClear={handleClear}
              onCancel={handleCancel}
              openSettings={() => setSettingsOpen(true)}
              error={factCheckError}
            />
          ) : (
            <>
              {/* Hero Section - Editorial, not template */}
              <div className="text-center space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
                {/* Dynamic Visual Anchor - the animation */}
                <div className="relative mb-4">
                  <DashboardHero />
                </div>

                {/* Headline - Bold, dramatic typography */}
                <div className="space-y-3">
                  <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight">
                    <span className="bg-gradient-to-br from-foreground via-foreground to-muted-foreground bg-clip-text text-transparent">
                      Truth, Verified.
                    </span>
                  </h1>
                  <p className="text-lg text-muted-foreground max-w-xl mx-auto leading-relaxed">
                    AI-powered analysis for text, images, and video deepfakes. Paste a URL or upload
                    content to begin.
                  </p>
                </div>
              </div>

              {/* Composer Input - The main action area */}
              <div className="w-full animate-in fade-in slide-in-from-bottom-6 duration-700 delay-100">
                <InputCard
                  input={input}
                  setInput={setInput}
                  loading={loading}
                  onFactCheck={handleFactCheck}
                  onClear={handleClear}
                  textSize={prefs.textSize}
                  openSettings={() => setSettingsOpen(true)}
                  onAIDetection={handleAIDetection}
                  onInputTypeChange={handleInputTypeChange}
                  saveImageToHistory={saveImageToHistory}
                  saveVideoToHistory={saveVideoToHistory}
                />
              </div>

              {/* Quick Action Chips */}
              <div className="flex flex-wrap justify-center gap-3 animate-in fade-in duration-700 delay-200">
                {quickActions.map((action) => (
                  <button
                    key={action.label}
                    onClick={() => handleQuickAction(action.label)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-muted hover:bg-muted/80 border border-border hover:border-primary/30 transition-all duration-200 group"
                  >
                    <action.icon className={cn('h-4 w-4', action.color)} />
                    <span className="text-sm font-medium text-muted-foreground group-hover:text-foreground transition-colors">
                      {action.label}
                    </span>
                  </button>
                ))}
              </div>

              {/* Recent Activity Section */}
              {history.length > 0 && (
                <div className="mt-8 space-y-4 animate-in fade-in duration-700 delay-300">
                  <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
                    <span className="inline-block w-1 h-5 bg-primary rounded-full" />
                    Recent Activity
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {history.slice(0, 3).map((item) => (
                      <RecentActivityCard key={item.id} item={item} />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
