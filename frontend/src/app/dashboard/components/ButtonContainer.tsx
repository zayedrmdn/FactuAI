import { RotateCcw, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function ButtonContainer({
  onSample,
  onDemo,
}: {
  onSample: () => void;
  onDemo: () => void;
}) {
  return (
    <div className="mt-8 flex flex-wrap justify-center gap-3 animate-in fade-in duration-1000 delay-150">
      <p className="w-full text-center text-xs font-medium text-muted-foreground uppercase tracking-widest mb-2">
        Try a specific scenario
      </p>
      <Button
        variant="outline"
        onClick={onSample}
        className="rounded-full bg-background/50 hover:bg-background border-primary/20 hover:border-primary/50 text-muted-foreground hover:text-foreground transition-all duration-300"
      >
        <RotateCcw className="h-3.5 w-3.5 mr-2" />
        Medical Misinfo
      </Button>
      <Button
        variant="outline"
        onClick={onDemo}
        className="rounded-full bg-background/50 hover:bg-background border-primary/20 hover:border-primary/50 text-muted-foreground hover:text-foreground transition-all duration-300"
      >
        <Sparkles className="h-3.5 w-3.5 mr-2" />
        Conspiracy Theory
      </Button>
    </div>
  );
}
