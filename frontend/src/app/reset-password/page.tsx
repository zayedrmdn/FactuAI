"use client";

import { useState, useEffect, Suspense } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { motion } from "framer-motion";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { 
  ArrowLeft, 
  Lock, 
  Eye, 
  EyeOff, 
  CheckCircle, 
  AlertCircle, 
  Shield 
} from "lucide-react";

// Form validation schema
const resetPasswordSchema = z.object({
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      "Password must contain at least one uppercase letter, one lowercase letter, and one number"
    ),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

type ResetPasswordForm = z.infer<typeof resetPasswordSchema>;

function ResetPasswordContent() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [tokenValid, setTokenValid] = useState<boolean | null>(null);

  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get("token");

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<ResetPasswordForm>({
    resolver: zodResolver(resetPasswordSchema),
  });

  const password = watch("password");

  // Validate token on component mount
  useEffect(() => {
    if (!token) {
      setTokenValid(false);
      return;
    }

    // You could add a token validation endpoint here
    // For now, we'll assume the token is valid if present
    setTokenValid(true);
  }, [token]);

  const onSubmit = async (data: ResetPasswordForm) => {
    if (!token) {
      setError("Invalid reset token");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/api/auth/reset-password", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          token,
          new_password: data.password,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setIsSuccess(true);
        // Redirect to login after 3 seconds
        setTimeout(() => {
          router.push("/login?message=Password reset successful");
        }, 3000);
      } else {
        setError(result.message || "An error occurred. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please check your connection and try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Invalid token state
  if (tokenValid === false) {
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
              This password reset link has expired or is invalid
            </p>
          </motion.div>
        </div>

        {/* Right Side - Error Message */}
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
                className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-6"
              >
                <AlertCircle className="w-8 h-8 text-red-500" />
              </motion.div>

              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Invalid Reset Link
              </h2>

              <p className="text-gray-600 dark:text-gray-300 mb-6">
                This password reset link is invalid or has expired. Please request a new one.
              </p>

              <div className="space-y-3">
                <Link
                  href="/forgot-password"
                  className="block w-full px-6 py-3 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-lg transition-all duration-200"
                >
                  Request New Reset Link
                </Link>
                
                <Link
                  href="/login"
                  className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
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

  // Success state
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
              Your password has been successfully updated
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
                Password Reset Successfully!
              </h2>

              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Your password has been updated. You'll be redirected to the login page shortly.
              </p>

              <Link
                href="/login"
                className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-lg transition-all duration-200"
              >
                Continue to Login
              </Link>
            </div>
          </motion.div>
        </div>
      </main>
    );
  }

  // Loading state
  if (tokenValid === null) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-blue-600/30 border-t-blue-600 rounded-full animate-spin" />
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
            Create a new secure password for your account
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
                <Shield className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Reset Your Password
              </h2>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Enter your new password below
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
              {/* New Password */}
              <div>
                <div className="relative">
                  <input
                    {...register("password")}
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter new password"
                    className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    disabled={isSubmitting}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.password && (
                  <p className="text-sm text-red-500 mt-2">{errors.password.message}</p>
                )}
              </div>

              {/* Confirm Password */}
              <div>
                <div className="relative">
                  <input
                    {...register("confirmPassword")}
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="Confirm new password"
                    className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    disabled={isSubmitting}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="text-sm text-red-500 mt-2">{errors.confirmPassword.message}</p>
                )}
              </div>

              {/* Password Requirements */}
              {password && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-xl border border-gray-200 dark:border-gray-600"
                >
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Password Requirements:
                  </h4>
                  <div className="space-y-1 text-xs">
                    <div
                      className={`flex items-center gap-2 ${
                        password.length >= 8
                          ? "text-green-600 dark:text-green-400"
                          : "text-gray-500 dark:text-gray-400"
                      }`}
                    >
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${
                          password.length >= 8 ? "bg-green-500" : "bg-gray-300"
                        }`}
                      />
                      At least 8 characters
                    </div>
                    <div
                      className={`flex items-center gap-2 ${
                        /[A-Z]/.test(password)
                          ? "text-green-600 dark:text-green-400"
                          : "text-gray-500 dark:text-gray-400"
                      }`}
                    >
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${
                          /[A-Z]/.test(password) ? "bg-green-500" : "bg-gray-300"
                        }`}
                      />
                      One uppercase letter
                    </div>
                    <div
                      className={`flex items-center gap-2 ${
                        /[a-z]/.test(password)
                          ? "text-green-600 dark:text-green-400"
                          : "text-gray-500 dark:text-gray-400"
                      }`}
                    >
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${
                          /[a-z]/.test(password) ? "bg-green-500" : "bg-gray-300"
                        }`}
                      />
                      One lowercase letter
                    </div>
                    <div
                      className={`flex items-center gap-2 ${
                        /\d/.test(password)
                          ? "text-green-600 dark:text-green-400"
                          : "text-gray-500 dark:text-gray-400"
                      }`}
                    >
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${
                          /\d/.test(password) ? "bg-green-500" : "bg-gray-300"
                        }`}
                      />
                      One number
                    </div>
                  </div>
                </motion.div>
              )}

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full px-6 py-3 bg-neutral-900 hover:bg-neutral-800 disabled:bg-neutral-400 text-white font-medium rounded-lg transition-all duration-200 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
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

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-blue-600/30 border-t-blue-600 rounded-full animate-spin" />
      </div>
    }>
      <ResetPasswordContent />
    </Suspense>
  );
}
