'use client';

import { motion } from 'framer-motion';
import { Plus, X, Circle, Box, Triangle, Hexagon, type LucideIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

// Geometric shapes to float in background
const shapes = [
  { Icon: Plus, size: 24, delay: 0 },
  { Icon: Circle, size: 16, delay: 2 },
  { Icon: Box, size: 20, delay: 4 },
  { Icon: X, size: 18, delay: 1 },
  { Icon: Triangle, size: 22, delay: 3 },
  { Icon: Hexagon, size: 20, delay: 5 },
];

export function HeroAnimation() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none select-none">
      {/* 
        Background Gradient Mesh 
        Subtle moving gradients to give life to the void
      */}
      <motion.div
        className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] bg-primary/5 rounded-full blur-[100px]"
        animate={{
          x: [0, 50, 0],
          y: [0, 30, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
      />

      <motion.div
        className="absolute bottom-[10%] right-[5%] w-[40%] h-[40%] bg-blue-500/5 rounded-full blur-[120px]"
        animate={{
          x: [0, -40, 0],
          y: [0, -60, 0],
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{ duration: 25, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
      />

      {/* Floating Geometric Symbols */}
      {shapes.map((shape, i) => (
        <FloatingShape key={i} index={i} {...shape} />
      ))}
    </div>
  );
}

interface FloatingShapeProps {
  Icon: LucideIcon;
  size: number;
  index: number;
  delay: number;
}

function FloatingShape({ Icon, size, index, delay }: FloatingShapeProps) {
  const initialTop = `${(index * 17 + 10) % 80}%`;
  const initialLeft = `${(index * 23 + 5) % 90}%`;

  return (
    <motion.div
      className="absolute text-primary/10 dark:text-primary/20"
      style={{
        top: initialTop,
        left: initialLeft,
      }}
      initial={{ opacity: 0, scale: 0 }}
      animate={{
        opacity: [0.1, 0.3, 0.1],
        y: [0, -40, 0],
        rotate: [0, 180, 360],
        scale: [1, 1.1, 1],
      }}
      transition={{
        duration: 15 + index * 2,
        repeat: Infinity,
        ease: 'easeInOut',
        delay,
      }}
    >
      <Icon size={size} strokeWidth={1.5} />
    </motion.div>
  );
}
