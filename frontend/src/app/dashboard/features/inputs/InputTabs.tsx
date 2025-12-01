import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Image as ImageIcon, Video } from 'lucide-react';
import TextTab from './TextInput/TextTab';
import ImageTab from './ImageInput/ImageTab';
import VideoTab from './VideoInput/VideoTab';
import { InputType, TextSize } from '../../types/ui';
import { validateBasic } from '../../utils/validation';
import { cn } from "@/lib/utils";

interface InputTabsProps {
  input: string;
  setInput: (value: string) => void;
  textSize: TextSize;
  onClear: () => void;
  onImageProcessed: (text: string, aiScore: number | null, imageUrl: string, aiError?: string) => void;
  onVideoProcessed: (text: string, filename?: string, videoUrl?: string) => void;
  onInputTypeChange: (type: InputType) => void;
}

const tabs = [
  { id: 'text' as InputType, label: 'Text', icon: FileText },
  { id: 'image' as InputType, label: 'Image', icon: ImageIcon },
  { id: 'video' as InputType, label: 'Video', icon: Video },
];

export default function InputTabs({
  input,
  setInput,
  textSize,
  onClear,
  onImageProcessed,
  onVideoProcessed,
  onInputTypeChange
}: InputTabsProps) {
  const [activeTab, setActiveTab] = useState<InputType>('text');
  const validationResult = validateBasic(input);

  const handleTabChange = (tabId: InputType) => {
    setActiveTab(tabId);
    onInputTypeChange(tabId);
  };

  return (
    <div className="w-full">
      {/* Enhanced Tab Headers */}
      <div className="flex p-1 bg-muted/60 rounded-lg border border-border/40">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={cn(
                "relative flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-all duration-200",
                isActive
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-background/50"
              )}
            >
              <Icon className={cn("w-4 h-4", isActive ? "text-primary" : "text-muted-foreground")} />
              <span className="relative z-10">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Enhanced Tab Content */}
      <div className="pt-6">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'text' && (
            <TextTab
              input={input}
              setInput={setInput}
              textSize={textSize}
              onClear={onClear}
            />
          )}
          
          {activeTab === 'image' && (
            <ImageTab onImageProcessed={onImageProcessed} />
          )}
          
          {activeTab === 'video' && (
            <VideoTab onVideoProcessed={onVideoProcessed} />
          )}
        </motion.div>
      </div>
    </div>
  );
}
