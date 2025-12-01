import React, { useState, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ShieldCheckIcon, Cog6ToothIcon } from "@heroicons/react/24/outline";
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
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-xl font-semibold">FactuAI</CardTitle>
        <button
          onClick={openSettings}
          className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100"
          aria-label="Settings"
        >
          <Cog6ToothIcon className="w-5 h-5" />
        </button>
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
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={onFactCheck}
              disabled={loading !== null || !validationResult.isValid}
              className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  {loading === "summary" ? "Summarizing..." : "Fact-checking..."}
                </>
              ) : (
                <>
                  <ShieldCheckIcon className="w-5 h-5" />
                  Fact-Check
                </>
              )}
            </button>

            <button
              onClick={handleClear}
              disabled={loading !== null}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
            >
              Clear
            </button>
          </div>
        )}

        {showValidationError && (
          <div className="text-sm text-red-600 bg-red-50 p-3 rounded-lg">
            <p className="font-medium">{validationResult.error}</p>
            {validationResult.suggestion && (
              <p className="text-xs text-red-500 mt-1">{validationResult.suggestion}</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
