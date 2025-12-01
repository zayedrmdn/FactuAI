"use client";

import { useState, useCallback } from "react";

export function useInputType() {
  const [currentInputType, setCurrentInputType] = useState<"text" | "image" | "video">("text");
  const [currentImageData, setCurrentImageData] = useState<{
    url: string;
    aiScore: number | null;
    aiError?: string;
  } | null>(null);
  const [currentVideoData, setCurrentVideoData] = useState<{
    filename: string;
    videoUrl?: string;
  } | null>(null);

  // Input type change handler
  const handleInputTypeChange = useCallback((
    type: "text" | "image" | "video",
    data?: {
      imageData?: {url: string, aiScore: number | null, aiError?: string};
      videoData?: {filename: string, videoUrl?: string};
    }
  ) => {
    setCurrentInputType(type);
    
    if (data?.imageData) {
      setCurrentImageData(data.imageData);
    } else if (type !== "image") {
      setCurrentImageData(null);
    }
    
    if (data?.videoData) {
      setCurrentVideoData(data.videoData);
    } else if (type !== "video") {
      setCurrentVideoData(null);
    }
  }, []);

  // AI detection handler
  const handleAIDetection = useCallback((score: number | null, error?: string) => {
    if (score !== null || error) {
      setCurrentInputType("image");
      setCurrentImageData({
        url: "",
        aiScore: score,
        aiError: error
      });
    }
  }, []);

  // Clear input type data
  const clearInputTypeData = useCallback(() => {
    setCurrentInputType("text");
    setCurrentImageData(null);
    setCurrentVideoData(null);
  }, []);

  return {
    currentInputType,
    currentImageData,
    currentVideoData,
    handleInputTypeChange,
    handleAIDetection,
    clearInputTypeData,
  };
}
