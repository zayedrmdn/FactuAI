"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import Link from "next/link";

const schema = z
  .object({
    username: z.string().min(2, "Username must be at least 2 characters").optional(),
    email: z.string().email("Invalid email"),
    password: z.string().min(6, "Password must be at least 6 characters"),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  });

type FormData = z.infer<typeof schema>;

export default function RegisterPage() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  // Triggered when the user submits the registration form
const onSubmit = async (data: FormData) => {
  try {
    // Send a POST request to the backend with the registration data
    const res = await fetch("http://127.0.0.1:5000/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: data.username,
        email: data.email,
        password: data.password,
      }),
    });

    // Convert the response to JSON format
    const result = await res.json();

    if (res.ok) {
      // Display a success message if registration is successful
      toast.success(result.message || "Registration successful");

      // Redirect the user to the login page after a short delay
      setTimeout(() => {
        window.location.href = "/login";
      }, 1000);
    } else {
      // Display an error message if the server returns a failure response
      toast.error(result.error || "Registration failed");
    }
  } catch (err: any) {
    // Handle unexpected runtime errors and notify the user
    console.error(err);
    toast.error("Something went wrong");
  }
};


  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex">
      {/* Left Side - Brand & Illustration */}
      <div className="hidden lg:flex lg:flex-1 flex-col items-center justify-center p-12 bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-emerald-900/20 dark:to-teal-900/20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          {/* Logo */}
          <div className="w-20 h-20 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-xl">
            <span className="text-3xl">🔍</span>
          </div>
          
          {/* Brand */}
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4 tracking-tight">
            Join FactuAI
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-md">
            Start your journey with intelligent fact-checking. Verify news, combat misinformation.
          </p>
          
          {/* Abstract Illustration */}
          <div className="relative">
            <svg
              className="w-64 h-64 text-emerald-300 dark:text-emerald-600"
              fill="currentColor"
              viewBox="0 0 200 200"
            >
              <circle cx="40" cy="60" r="18" opacity="0.6" />
              <circle cx="120" cy="40" r="22" opacity="0.4" />
              <circle cx="80" cy="100" r="28" opacity="0.8" />
              <circle cx="160" cy="120" r="20" opacity="0.5" />
              <circle cx="60" cy="160" r="16" opacity="0.6" />
              <circle cx="140" cy="170" r="24" opacity="0.4" />
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
              <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <span className="text-xl text-white">🔍</span>
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Register
              </h2>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Create your account to start verifying news instantly.
              </p>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
              <div>
                <input
                  type="text"
                  placeholder="Username (optional)"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all duration-200"
                  {...register("username")}
                />
                {errors.username && (
                  <p className="text-sm text-red-500 mt-2">{errors.username.message}</p>
                )}
              </div>

              <div>
                <input
                  type="email"
                  placeholder="Email"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all duration-200"
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
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all duration-200"
                  {...register("password")}
                />
                {errors.password && (
                  <p className="text-sm text-red-500 mt-2">{errors.password.message}</p>
                )}
              </div>

              <div>
                <input
                  type="password"
                  placeholder="Confirm Password"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all duration-200"
                  {...register("confirmPassword")}
                />
                {errors.confirmPassword && (
                  <p className="text-sm text-red-500 mt-2">
                    {errors.confirmPassword.message}
                  </p>
                )}
              </div>

              <Button 
                type="submit" 
                className="w-full bg-neutral-900 hover:bg-neutral-800 active:scale-95 transition-all duration-200 py-3 text-white font-medium rounded-lg"
              >
                Register
              </Button>

              <div className="text-center pt-4">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Already have an account?{" "}
                  <Link 
                    href="/login" 
                    className="text-emerald-600 dark:text-emerald-400 font-semibold hover:text-emerald-700 dark:hover:text-emerald-300 hover:underline transition-colors duration-200"
                  >
                    Login here
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