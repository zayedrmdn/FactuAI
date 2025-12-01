import React, { useState, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ShieldCheck, Settings, Loader2, X } from "lucide-react";
import { validateBasic } from "../../utils/validation";
import InputTabs from "./InputTabs";
import { InputType, TextSize } from "../../types/ui";

interface InputCardProps {
  input: string;
  setInput: (v: string) => void;
  loading: null | "summary" | "factcheck";
  onFactCheck: () => void;
  onClear: () => void;
  textSize: TextSize;
  openSettings: () => void;
  onAIDetection?: (score: number | null, error?: string) => void;
  onInputTypeChange?: (
    type: InputType,
    data?: { imageData?: {url: string, aiScore: number | null, aiError?: string}, videoData?: {filename: string, videoUrl?: string} }
  ) => void;
  saveImageToHistory?: (text: string, imageUrl: string, aiScore: number | null, aiError?: string) => void;
  saveVideoToHistory?: (text: string, filename: string, videoUrl?: string) => void;
}

export default function InputCard({
  input,
  setInput,
  loading,
  onFactCheck,
  onClear,
  textSize,
  openSettings,
  onAIDetection,
  onInputTypeChange,
  saveImageToHistory,
  saveVideoToHistory,
}: InputCardProps) {
  const [currentInputType, setCurrentInputType] = useState<InputType>("text");

  const validationResult = validateBasic(input);
  const showValidationError = input.trim().length > 0 && validationResult.error;

  const handleImageProcessed = useCallback((
    text: string, 
    aiScore: number | null, 
    imageUrl: string,
    aiError?: string
  ) => {
    setInput(text);
    setCurrentInputType("image");
    
    const imageData = { url: imageUrl, aiScore, aiError };

    if (onAIDetection) {
      onAIDetection(aiScore, aiError);
    }

    if (onInputTypeChange) {
      onInputTypeChange("image", { imageData });
    }

    if (saveImageToHistory) {
      saveImageToHistory(text, imageUrl, aiScore, aiError);
    }
  }, [setInput, onAIDetection, onInputTypeChange, saveImageToHistory]);

  const handleVideoProcessed = useCallback((text: string, filename?: string, videoUrl?: string) => {
    setInput(text);
    setCurrentInputType("video");
    
    const videoData = { 
      filename: filename || "Unknown video",
      videoUrl: videoUrl
    };

    if (onInputTypeChange) {
      onInputTypeChange("video", { videoData });
    }

    if (saveVideoToHistory && text.trim() && filename) {
      saveVideoToHistory(text, filename, videoUrl);
    }
  }, [setInput, onInputTypeChange, saveVideoToHistory]);

  const handleTextInput = useCallback((text: string) => {
    setInput(text);
    setCurrentInputType("text");

    if (onInputTypeChange) {
      onInputTypeChange("text");
    }
  }, [setInput, onInputTypeChange]);

  const handleInputTypeChange = useCallback((type: InputType) => {
    setCurrentInputType(type);
  }, []);

  const handleClear = useCallback(() => {
    setInput("");
    setCurrentInputType("text");
    onClear();
  }, [setInput, onClear]);

  return (
    <Card className="w-full shadow-sm border-border/60">
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-6">
        <div className="space-y-1">
          <CardTitle className="text-xl font-semibold tracking-tight text-foreground">
            Investigation Console
          </CardTitle>
          <CardDescription>
            Analyze text, images, or videos to verify claims and detect manipulation.
          </CardDescription>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={openSettings}
          className="text-muted-foreground hover:text-foreground"
          aria-label="Settings"
        >
          <Settings className="h-5 w-5" />
        </Button>
      </CardHeader>

      <CardContent className="space-y-6">
        <InputTabs
          input={input}
          setInput={handleTextInput}
          textSize={textSize}
          onClear={handleClear}
          onImageProcessed={handleImageProcessed}
          onVideoProcessed={handleVideoProcessed}
          onInputTypeChange={handleInputTypeChange}
        />

        {input && (
          <div className="flex flex-col gap-3 sm:flex-row">
            <Button
              onClick={onFactCheck}
              disabled={loading !== null || !validationResult.isValid}
              className="flex-1 gap-2"
              size="lg"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {loading === "summary" ? "Summarizing..." : "Fact-checking..."}
                </>
              ) : (
                <>
                  <ShieldCheck className="h-4 w-4" />
                  Verify Claim
                </>
              )}
            </Button>

            <Button
              variant="outline"
              onClick={handleClear}
              disabled={loading !== null}
              size="lg"
              className="gap-2"
            >
              <X className="h-4 w-4" />
              Clear
            </Button>
          </div>
        )}

        {showValidationError && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            <p className="font-medium">{validationResult.error}</p>
            {validationResult.suggestion && (
              <p className="mt-1 text-xs opacity-90">{validationResult.suggestion}</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
