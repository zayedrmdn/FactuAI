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
      {/* Tab Content First - padded container keeps media tools contained */}
      <div className="px-4 pt-4 sm:px-6 sm:pt-6">
        <div className="relative min-h-[200px] rounded-xl border border-border/60 bg-muted/10">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.15 }}
            className="p-4 sm:p-6"
          >
            {activeTab === 'text' && (
              <TextInput input={input} setInput={setInput} textSize={textSize} onClear={onClear} />
            )}

            {activeTab === 'image' && <ImageInput onImageProcessed={onImageProcessed} />}

            {activeTab === 'video' && <VideoInput onVideoProcessed={onVideoProcessed} />}
          </motion.div>
        </div>
      </div>

      {/* Bottom Toolbar with Tab Buttons + Format Info */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3 sm:px-6 border-t border-border/50 bg-card/60 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={cn(
                'p-2 rounded-lg transition-all duration-200',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              )}
              title={tab.label}
            >
              <Icon className="w-5 h-5" />
            </button>
          );
        })}

        <div className="h-5 w-px bg-border mx-2" />

        <span className="text-xs text-muted-foreground font-medium hidden sm:block">
          Supports JPG, PNG, MP4, TXT
        </span>
      </div>
    </div>
  );
}
