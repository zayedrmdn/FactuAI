'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

export function DashboardHero() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="relative w-64 h-64 md:w-80 md:h-80 mx-auto flex items-center justify-center select-none pointer-events-none perspective-1000">
      {/* 
        Core Glow - Deep atmospheric lighting
        Using primary/20 for a subtle, high-tech feel that works in both light/dark modes
      */}
      <div className="absolute inset-0 bg-primary/20 blur-[80px] rounded-full opacity-40 dark:opacity-20 animate-pulse-slow" />

      {/* 
        Outer Ring - The "Scanner"
        Slow counter-clockwise rotation with variable opacity
      */}
      <motion.div
        className="absolute inset-0 border border-primary/10 rounded-full"
        animate={{ rotate: -360 }}
        transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-primary/20 rounded-full blur-[1px]" />
      </motion.div>

      {/* 
        Middle Ring - "Processing Layer" 
        Dashed border for technical texture, clockwise rotation
      */}
      <motion.div
        className="absolute inset-8 border border-dashed border-primary/20 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 45, repeat: Infinity, ease: 'linear' }}
      />

      {/* 
        Inner Ring - "Focus Field"
        Eccentric orbit for organic feel
      */}
      <motion.div
        className="absolute inset-16 border border-primary/30 rounded-full opacity-60"
        animate={{
          rotate: [0, 360],
          scale: [1, 1.05, 1],
        }}
        transition={{
          rotate: { duration: 20, repeat: Infinity, ease: 'linear' },
          scale: { duration: 4, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-primary/60 rounded-full" />
      </motion.div>

      {/* 
        The Core - "FactuAI Logic Engine"
        Geometric abstraction of a processing unit
      */}
      <div className="relative w-24 h-24 grid place-items-center">
        {/* Central Square - Stable anchor */}
        <motion.div
          className="absolute w-12 h-12 bg-primary/10 border border-primary/30 rotate-45 backdrop-blur-sm"
          animate={{ rotate: [45, 225] }}
          transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
        />

        {/* Overlapping Square - Dynamic layer */}
        <motion.div
          className="absolute w-12 h-12 border border-primary/20 rotate-45"
          animate={{ rotate: [45, -135] }}
          transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
        />

        {/* The Heart - Verified pulse */}
        <motion.div
          className="relative w-4 h-4 bg-primary rounded-sm shadow-lg shadow-primary/50"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.8, 1, 0.8],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* Floating Particles - "Data points" */}
      {[...Array(6)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-primary/40 rounded-full"
          initial={{ opacity: 0, scale: 0 }}
          animate={{
            opacity: [0, 1, 0],
            scale: [0, 1.5, 0],
            x: Math.cos(i * 60 * (Math.PI / 180)) * 100,
            y: Math.sin(i * 60 * (Math.PI / 180)) * 100,
          }}
          transition={{
            duration: 3,
            delay: i * 0.5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}
