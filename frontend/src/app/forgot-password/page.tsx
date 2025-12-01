"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowLeft, Mail, CheckCircle, AlertCircle } from "lucide-react";

// Form validation schema
const forgotPasswordSchema = z.object({
  email: z.string().email("Please enter a valid email address"),
});

type ForgotPasswordForm = z.infer<typeof forgotPasswordSchema>;

export default function ForgotPasswordPage() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    getValues,
  } = useForm<ForgotPasswordForm>({
    resolver: zodResolver(forgotPasswordSchema),
  });

  const onSubmit = async (data: ForgotPasswordForm) => {
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/api/auth/request-password-reset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (response.ok) {
        setIsSuccess(true);
      } else {
        setError(result.message || "An error occurred. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please check your connection and try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isSuccess) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex">
        {/* Left Side - Brand & Illustration */}
        <div className="hidden lg:flex lg:flex-1 flex-col items-center justify-center p-12 bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            {/* Logo */}
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-xl">
              <span className="text-3xl">🔍</span>
            </div>
            
            {/* Brand */}
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4 tracking-tight">
              FactuAI
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-md">
              Check your email for the password reset link
            </p>
          </motion.div>
        </div>

        {/* Right Side - Success Message */}
        <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="w-full max-w-md"
          >
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8 border border-gray-200 dark:border-gray-700 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-6"
              >
                <CheckCircle className="w-8 h-8 text-green-500" />
              </motion.div>

              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Check Your Email
              </h2>

              <p className="text-gray-600 dark:text-gray-300 mb-6 leading-relaxed">
                We've sent a password reset link to{" "}
                <span className="font-medium text-gray-900 dark:text-white">
                  {getValues("email")}
                </span>
                . Click the link in the email to reset your password.
              </p>

              <div className="space-y-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Didn't receive the email? Check your spam folder or try again.
                </div>

                <Link
                  href="/login"
                  className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-lg transition-all duration-200"
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

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex">
      {/* Left Side - Brand & Illustration */}
      <div className="hidden lg:flex lg:flex-1 flex-col items-center justify-center p-12 bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          {/* Logo */}
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-xl">
            <span className="text-3xl">🔍</span>
          </div>
          
          {/* Brand */}
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4 tracking-tight">
            FactuAI
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-md">
            Secure password recovery for your trusted fact-checking companion
          </p>
          
          {/* Abstract Illustration */}
          <div className="relative">
            <svg
              className="w-64 h-64 text-blue-300 dark:text-blue-600"
              fill="currentColor"
              viewBox="0 0 200 200"
            >
              <circle cx="50" cy="50" r="20" opacity="0.6" />
              <circle cx="150" cy="80" r="25" opacity="0.4" />
              <circle cx="100" cy="140" r="15" opacity="0.8" />
              <path d="M50 50 Q100 20 150 80" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.3" />
              <path d="M150 80 Q120 110 100 140" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.3" />
            </svg>
          </div>
        </motion.div>
      </div>

      {/* Right Side - Reset Form */}
      <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-md"
        >
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8 border border-gray-200 dark:border-gray-700">
            {/* Visual Anchor */}
            <div className="text-center mb-8">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Mail className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Forgot Password?
              </h2>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Enter your email and we'll send you a reset link
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3"
              >
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-red-700 dark:text-red-300">
                  {error}
                </div>
              </motion.div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div>
                <input
                  {...register("email")}
                  type="email"
                  placeholder="Enter your email address"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  disabled={isSubmitting}
                />
                {errors.email && (
                  <p className="text-sm text-red-500 mt-2">{errors.email.message}</p>
                )}
              </div>

              <button
                type="submit" 
                disabled={isSubmitting}
                className="w-full bg-neutral-900 hover:bg-neutral-800 active:scale-95 transition-all duration-200 py-3 text-white font-medium rounded-lg flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Sending Reset Link...
                  </>
                ) : (
                  <>
                    <Mail className="w-4 h-4" />
                    Send Reset Link
                  </>
                )}
              </button>

              <div className="text-center pt-4">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Remember your password?{" "}
                  <Link 
                    href="/login" 
                    className="text-blue-600 dark:text-blue-400 font-semibold hover:text-blue-700 dark:hover:text-blue-300 hover:underline transition-colors duration-200"
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
