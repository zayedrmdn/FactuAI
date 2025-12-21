'use client';

import { useState, useEffect, Suspense, type ReactNode } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeft,
  Lock,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  Shield,
  Search,
} from 'lucide-react';

// Form validation schema
const resetPasswordSchema = z
  .object({
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
        'Password must contain at least one uppercase letter, one lowercase letter, and one number'
      ),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ['confirmPassword'],
  });

type ResetPasswordForm = z.infer<typeof resetPasswordSchema>;

/**
 * Brand panel component for left side of split layout
 */
function BrandPanel({ message, children }: Readonly<{ message: string; children?: ReactNode }>) {
  return (
    <div className="hidden lg:flex lg:flex-1 flex-col items-center justify-center p-12 bg-gradient-to-br from-primary/10 to-info/10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center"
      >
        <div className="w-20 h-20 bg-primary text-primary-foreground rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-xl">
          <Search className="w-10 h-10" />
        </div>
        <h1 className="text-4xl font-bold text-foreground mb-4 tracking-tight">FactuAI</h1>
        <p className="text-xl text-muted-foreground mb-8 max-w-md">{message}</p>
        {children}
      </motion.div>
    </div>
  );
}

/**
 * Invalid token state component
 */
function InvalidTokenView() {
  return (
    <main className="min-h-screen bg-background flex">
      <BrandPanel message="This password reset link has expired or is invalid" />
      <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-md"
        >
          <div className="bg-card text-card-foreground rounded-2xl shadow-2xl p-8 border border-border text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mx-auto mb-6"
            >
              <AlertCircle className="w-8 h-8 text-destructive" />
            </motion.div>
            <h2 className="text-3xl font-bold mb-4">Invalid Reset Link</h2>
            <p className="text-muted-foreground mb-6">
              This password reset link is invalid or has expired. Please request a new one.
            </p>
            <div className="space-y-3">
              <Link
                href="/forgot-password"
                className="block w-full px-6 py-3 bg-primary hover:bg-primary/90 text-primary-foreground font-medium rounded-lg transition-all duration-200"
              >
                Request New Reset Link
              </Link>
              <Link
                href="/login"
                className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors duration-200"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Login
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </main>
  );
}

/**
 * Success state component
 */
function SuccessView() {
  return (
    <main className="min-h-screen bg-background flex">
      <BrandPanel message="Your password has been successfully updated" />
      <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-md"
        >
          <div className="bg-card text-card-foreground rounded-2xl shadow-2xl p-8 border border-border text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="w-16 h-16 bg-success/15 rounded-full flex items-center justify-center mx-auto mb-6"
            >
              <CheckCircle className="w-8 h-8 text-success" />
            </motion.div>
            <h2 className="text-3xl font-bold mb-4">Password Reset Successfully!</h2>
            <p className="text-muted-foreground mb-6">
              Your password has been updated. You&apos;ll be redirected to the login page shortly.
            </p>
            <Link
              href="/login"
              className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-primary hover:bg-primary/90 text-primary-foreground font-medium rounded-lg transition-all duration-200"
            >
              Continue to Login
            </Link>
          </div>
        </motion.div>
      </div>
    </main>
  );
}

/**
 * Loading state component
 */
function LoadingView() {
  return (
    <main className="min-h-screen bg-background flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
    </main>
  );
}

/**
 * Password requirement indicator
 */
function RequirementItem({ met, label }: Readonly<{ met: boolean; label: string }>) {
  const textClass = met ? 'text-success' : 'text-muted-foreground';
  const dotClass = met ? 'bg-success' : 'bg-border';

  return (
    <div className={`flex items-center gap-2 ${textClass}`}>
      <div className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
      {label}
    </div>
  );
}

function ResetPasswordContent() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [tokenValid, setTokenValid] = useState<boolean | null>(null);

  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get('token');

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<ResetPasswordForm>({
    resolver: zodResolver(resetPasswordSchema),
  });

  const password = watch('password');

  // Validate token on component mount
  useEffect(() => {
    setTokenValid(Boolean(token));
  }, [token]);

  const onSubmit = async (data: ResetPasswordForm) => {
    if (!token) {
      setError('Invalid reset token');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, new_password: data.password }),
      });

      const result = await response.json();

      if (response.ok) {
        setIsSuccess(true);
        setTimeout(() => {
          router.push('/login?message=Password reset successful');
        }, 3000);
      } else {
        setError(result.message || 'An error occurred. Please try again.');
      }
    } catch (err) {
      console.error('Password reset request failed:', err);
      setError('Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle different states
  if (tokenValid === null) return <LoadingView />;
  if (tokenValid === false) return <InvalidTokenView />;
  if (isSuccess) return <SuccessView />;

  return (
    <main className="min-h-screen bg-background flex">
      {/* Left Side - Brand & Illustration */}
      <BrandPanel message="Create a new secure password for your account">
        <div className="relative">
          <svg className="w-64 h-64 text-primary/30" fill="currentColor" viewBox="0 0 200 200">
            <circle cx="50" cy="50" r="20" opacity="0.6" />
            <circle cx="150" cy="80" r="25" opacity="0.4" />
            <circle cx="100" cy="140" r="15" opacity="0.8" />
            <path
              d="M50 50 Q100 20 150 80"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              opacity="0.3"
            />
            <path
              d="M150 80 Q120 110 100 140"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              opacity="0.3"
            />
          </svg>
        </div>
      </BrandPanel>

      {/* Right Side - Reset Form */}
      <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-md"
        >
          <div className="bg-card text-card-foreground rounded-2xl shadow-2xl p-8 border border-border">
            {/* Visual Anchor */}
            <div className="text-center mb-8">
              <div className="w-12 h-12 bg-primary text-primary-foreground rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Shield className="w-6 h-6" />
              </div>
              <h2 className="text-3xl font-bold mb-2">Reset Your Password</h2>
              <p className="text-muted-foreground text-sm">Enter your new password below</p>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-xl flex items-start gap-3"
              >
                <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                <div className="text-sm text-destructive">{error}</div>
              </motion.div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {/* New Password */}
              <div>
                <div className="relative">
                  <input
                    {...register('password')}
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter new password"
                    className="w-full px-4 py-3 pr-12 rounded-lg border border-input bg-background text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background transition-all duration-200"
                    disabled={isSubmitting}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.password && (
                  <p className="text-sm text-destructive mt-2">{errors.password.message}</p>
                )}
              </div>

              {/* Confirm Password */}
              <div>
                <div className="relative">
                  <input
                    {...register('confirmPassword')}
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Confirm new password"
                    className="w-full px-4 py-3 pr-12 rounded-lg border border-input bg-background text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background transition-all duration-200"
                    disabled={isSubmitting}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="w-5 h-5" />
                    ) : (
                      <Eye className="w-5 h-5" />
                    )}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="text-sm text-destructive mt-2">{errors.confirmPassword.message}</p>
                )}
              </div>

              {/* Password Requirements */}
              {password && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="p-4 bg-muted rounded-xl border border-border"
                >
                  <h4 className="text-sm font-medium text-foreground mb-2">
                    Password Requirements:
                  </h4>
                  <div className="space-y-1 text-xs">
                    <RequirementItem met={password.length >= 8} label="At least 8 characters" />
                    <RequirementItem met={/[A-Z]/.test(password)} label="One uppercase letter" />
                    <RequirementItem met={/[a-z]/.test(password)} label="One lowercase letter" />
                    <RequirementItem met={/\d/.test(password)} label="One number" />
                  </div>
                </motion.div>
              )}

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full px-6 py-3 bg-primary hover:bg-primary/90 text-primary-foreground font-medium rounded-lg transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Updating Password...
                  </>
                ) : (
                  <>
                    <Lock className="w-4 h-4" />
                    Update Password
                  </>
                )}
              </button>

              <div className="text-center pt-4">
                <p className="text-sm text-muted-foreground">
                  Remember your password?{' '}
                  <Link
                    href="/login"
                    className="text-primary font-semibold hover:underline transition-colors duration-200"
                  >
                    Back to Login
                  </Link>
                </p>
              </div>
            </form>
          </div>
        </motion.div>
      </div>
    </main>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
        </div>
      }
    >
      <ResetPasswordContent />
    </Suspense>
  );
}
