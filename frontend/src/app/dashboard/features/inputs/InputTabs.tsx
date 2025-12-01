import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { DocumentTextIcon, PhotoIcon, VideoCameraIcon } from '@heroicons/react/24/outline';
import TextTab from './TextInput/TextTab';
import ImageTab from './ImageInput/ImageTab';
import VideoTab from './VideoInput/VideoTab';
import { InputType, TextSize } from '../../types/ui';
import { validateBasic } from '../../utils/validation';

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
  { id: 'text' as InputType, label: 'Text', icon: DocumentTextIcon },
  { id: 'image' as InputType, label: 'Image', icon: PhotoIcon },
  { id: 'video' as InputType, label: 'Video', icon: VideoCameraIcon },
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
    <div>
      {/* Enhanced Tab Headers */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 rounded-t-lg">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <motion.button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`relative flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-all duration-200 group ${
                isActive
                  ? 'border-blue-600 text-blue-600 bg-white dark:bg-gray-900 shadow-sm'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600 hover:bg-white/50 dark:hover:bg-gray-700/50'
              }`}
            >
              <motion.div
                animate={{ 
                  scale: isActive ? 1.1 : 1,
                  rotate: isActive ? [0, -10, 10, 0] : 0
                }}
                transition={{ 
                  scale: { duration: 0.2 },
                  rotate: { duration: 0.5 }
                }}
              >
                <Icon className="w-4 h-4" />
              </motion.div>
              <span className="relative">
                {tab.label}
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute -bottom-3 left-0 right-0 h-0.5 bg-blue-600 rounded-full"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </span>
            </motion.button>
          );
        })}
      </div>

      {/* Enhanced Tab Content */}
      <div className="pt-6">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
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
