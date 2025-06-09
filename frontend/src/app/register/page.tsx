"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const schema = z
  .object({
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

  const onSubmit = async (data: FormData) => {
    try {
      const res = await fetch("http://127.0.0.1:5000/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: data.email,
          password: data.password,
        }),
      });

      const result = await res.json();

      if (res.ok) {
        toast.success(result.message || "Registration successful");

        // ✅ Delay redirect to show toast
        setTimeout(() => {
          window.location.href = "/login";
        }, 1000);
      } else {
        toast.error(result.error || "Registration failed");
      }
    } catch (err: any) {
      console.error(err);
      toast.error("Something went wrong");
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center">
      <div className="w-full max-w-md p-8 border rounded-lg shadow">
        <h1 className="text-2xl font-semibold mb-6 text-center">Register</h1>
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
          <div>
            <input
              type="password"
              placeholder="Confirm Password"
              className="w-full px-4 py-2 border rounded"
              {...register("confirmPassword")}
            />
            {errors.confirmPassword && (
              <p className="text-sm text-red-500 mt-1">
                {errors.confirmPassword.message}
              </p>
            )}
          </div>
          <Button type="submit">Register</Button>
        </form>
      </div>
    </main>
  );
}
