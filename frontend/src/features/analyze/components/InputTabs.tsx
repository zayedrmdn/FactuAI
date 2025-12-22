import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Image as ImageIcon, Video } from 'lucide-react';
import TextInput from './TextInput';
import ImageInput from './ImageInput';
import VideoInput from './VideoInput';
import { InputType, TextSize } from '@/types/dashboard/ui';
import { cn } from '@/lib/utils';

interface InputTabsProps {
  input: string;
  setInput: (value: string) => void;
  textSize: TextSize;
  onClear: () => void;
  onImageProcessed: (
    text: string,
    aiScore: number | null,
    imageUrl: string,
    aiError?: string
  ) => void;
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
  onInputTypeChange,
}: Readonly<InputTabsProps>) {
  const [activeTab, setActiveTab] = useState<InputType>('text');

  const handleTabChange = (tabId: InputType) => {
    setActiveTab(tabId);
    onInputTypeChange(tabId);
  };

  return (
    <div className="w-full">
      {/* Enhanced Tab Headers - Minimal "Pill" Design */}
      <div className="flex items-center gap-2 p-1 mb-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={cn(
                'relative flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium rounded-full transition-all duration-300 ease-out',
                isActive
                  ? 'bg-primary/10 text-primary ring-1 ring-primary/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              )}
            >
              <Icon
                className={cn(
                  'w-4 h-4 transition-colors',
                  isActive ? 'text-primary' : 'text-muted-foreground/70'
                )}
              />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Enhanced Tab Content */}
      <div className="relative min-h-[160px]">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'text' && (
            <TextInput input={input} setInput={setInput} textSize={textSize} onClear={onClear} />
          )}

          {activeTab === 'image' && <ImageInput onImageProcessed={onImageProcessed} />}

          {activeTab === 'video' && <VideoInput onVideoProcessed={onVideoProcessed} />}
        </motion.div>
      </div>
    </div>
  );
}
