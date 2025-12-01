"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import Link from "next/link";

const schema = z.object({
  email: z.string().min(1, "Email or username is required"),
  password: z.string().min(6, "Password must be at least 6 characters"),
});

type FormData = z.infer<typeof schema>;

export default function LoginPage() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  const onSubmit = async (data: FormData) => {
    try {
      const res = await fetch("http://localhost:5000/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await res.json();

      if (!res.ok) {
        toast.error(result.error || "Invalid credentials");
        return;
      }

      toast.success("Login successful");
      localStorage.setItem("user", JSON.stringify(result.user));
      window.location.href = "/dashboard";
    } catch (err: any) {
      toast.error(err.message || "Something went wrong");
    }
  };

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
            Your trusted companion for intelligent fact-checking and news verification
          </p>
          
          {/* Abstract Illustration */}
          <div className="relative">
            <svg
              className="w-64 h-64 text-blue-300 dark:text-blue-600"
              fill="currentColor"
              viewBox="0 0 200 200"
            >
              <circle cx="50" cy="50" r="20" opacity="0.6" />
              <circle cx="150" cy="75" r="25" opacity="0.4" />
              <circle cx="100" cy="125" r="30" opacity="0.8" />
              <circle cx="75" cy="150" r="15" opacity="0.5" />
              <circle cx="175" cy="150" r="18" opacity="0.6" />
            </svg>
          </div>
        </motion.div>
      </div>

      {/* Right Side - Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="w-full max-w-md"
        >
          {/* Form Card */}
          <div className="bg-white dark:bg-neutral-900 rounded-xl shadow-lg px-10 py-8 border border-gray-100 dark:border-gray-800">
            {/* Visual Anchor */}
            <div className="text-center mb-8">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <span className="text-xl text-white">🔍</span>
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Login
              </h2>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Welcome back. Enter your credentials to continue.
              </p>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div>
                <input
                  type="text"
                  placeholder="Email or Username"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  {...register("email")}
                />
                {errors.email && (
                  <p className="text-sm text-red-500 mt-2">{errors.email.message}</p>
                )}
              </div>

              <div>
                <input
                  type="password"
                  placeholder="Password"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  {...register("password")}
                />
                {errors.password && (
                  <p className="text-sm text-red-500 mt-2">{errors.password.message}</p>
                )}
              </div>

              {/* Forgot Password Link */}
              <div className="text-right">
                <Link 
                  href="/forgot-password"
                  className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline transition-colors duration-200"
                >
                  Forgot your password?
                </Link>
              </div>

              <Button 
                type="submit" 
                className="w-full bg-neutral-900 hover:bg-neutral-800 active:scale-95 transition-all duration-200 py-3 text-white font-medium rounded-lg"
              >
                Log in
              </Button>

              <div className="text-center pt-4">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Don&apos;t have an account?{" "}
                  <Link 
                    href="/register" 
                    className="text-blue-600 dark:text-blue-400 font-semibold hover:text-blue-700 dark:hover:text-blue-300 hover:underline transition-colors duration-200"
                  >
                    Register here
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