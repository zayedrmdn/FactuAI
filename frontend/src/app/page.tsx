// Path: frontend/src/app/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  CallToAction,
  Features,
  Footer,
  HeroSection,
  HowItWorks,
  LandingNav,
} from '@/features/landing';

export default function HomePage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in - we no longer need to track this state
    // since we show landing page for all users
    setIsLoading(false);
  }, [router]);

  // Show loading while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  // Show landing page for all users (authenticated and non-authenticated)
  return (
    <div className="min-h-screen">
      <LandingNav />
      <HeroSection />
      <div id="features">
        <Features />
      </div>
      <HowItWorks />
      <CallToAction />
      <div id="contact">
        <Footer />
      </div>
    </div>
  );
}
