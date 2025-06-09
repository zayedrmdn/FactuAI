"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { toast } from "sonner"; // ✅ Use Sonner
import Link from "next/link";


const schema = z.object({
  email: z.string().email("Invalid email"),
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
    <main className="flex min-h-screen items-center justify-center">
      <div className="w-full max-w-md p-8 border rounded-lg shadow">
        <h1 className="text-2xl font-semibold mb-6 text-center">Login</h1>
        <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
          <div>
            <input
              type="email"
              placeholder="Email"
              className="w-full px-4 py-2 border rounded"
              {...register("email")}
            />
            {errors.email && (
              <p className="text-sm text-red-500 mt-1">{errors.email.message}</p>
            )}
          </div>
          <div>
            <input
              type="password"
              placeholder="Password"
              className="w-full px-4 py-2 border rounded"
              {...register("password")}
            />
            {errors.password && (
              <p className="text-sm text-red-500 mt-1">{errors.password.message}</p>
            )}
          </div>
          <Button type="submit">Log in</Button>
          <p className="text-center text-sm text-gray-600">
            Don&apos;t have an account?{" "}
            <Link href="/register" className="text-blue-600 hover:underline">
              Register here
            </Link>
          </p>
        </form>
      </div>
    </main>
  );
}
