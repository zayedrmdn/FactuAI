'use client';

import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/button';
import { Input, Label } from '@/components/ui/form-controls';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { toast } from 'sonner';
import Link from 'next/link';
import { ShieldCheck } from 'lucide-react';
import { getApiUrl } from '@/lib/apiBase';

const schema = z
  .object({
    username: z.string().min(2, 'Username must be at least 2 characters').optional(),
    email: z.string().email('Invalid email'),
    password: z.string().min(6, 'Password must be at least 6 characters'),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type FormData = z.infer<typeof schema>;

export default function RegisterPage() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  // Triggered when the user submits the registration form
  const onSubmit = async (data: FormData) => {
    try {
      // Send a POST request to the backend with the registration data
      const res = await fetch(getApiUrl('register'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        toast.success(result.message || 'Registration successful');

        // Redirect the user to the login page after a short delay
        setTimeout(() => {
          globalThis.location.href = '/login';
        }, 1000);
      } else {
        // Display an error message if the server returns a failure response
        toast.error(result.error || 'Registration failed');
      }
    } catch (err: unknown) {
      // Handle unexpected runtime errors and notify the user
      console.error(err);
      toast.error('Something went wrong');
    }
  };

  return (
    <div className="container relative min-h-screen flex-col items-center justify-center grid lg:max-w-none lg:grid-cols-2 lg:px-0 bg-background">
      <div className="relative hidden h-full flex-col bg-muted p-10 text-white lg:flex dark:border-r">
        <div className="absolute inset-0 bg-primary" />
        <div className="relative z-20 flex items-center text-lg font-medium">
          <ShieldCheck className="mr-2 h-6 w-6" />
          FactuAI
        </div>
        <div className="relative z-20 mt-auto">
          <blockquote className="space-y-2">
            <p className="text-lg">
              &ldquo;Start your journey with intelligent fact-checking. Verify news, combat
              misinformation, and trust what you read.&rdquo;
            </p>
            <footer className="text-sm">The FactuAI Team</footer>
          </blockquote>
        </div>
      </div>
      <div className="lg:p-8">
        <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
          <Card className="border-0 shadow-none sm:border sm:shadow-sm">
            <CardHeader className="space-y-1">
              <CardTitle className="text-2xl tracking-tight">Create an account</CardTitle>
              <CardDescription>Enter your email below to create your account</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <form onSubmit={handleSubmit(onSubmit)}>
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="username">Username</Label>
                    <Input
                      id="username"
                      placeholder="johndoe"
                      type="text"
                      autoCapitalize="none"
                      autoCorrect="off"
                      disabled={isSubmitting}
                      {...register('username')}
                    />
                    {errors.username && (
                      <p className="text-sm text-destructive">{errors.username.message}</p>
                    )}
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      placeholder="name@example.com"
                      type="email"
                      autoCapitalize="none"
                      autoComplete="email"
                      autoCorrect="off"
                      disabled={isSubmitting}
                      {...register('email')}
                    />
                    {errors.email && (
                      <p className="text-sm text-destructive">{errors.email.message}</p>
                    )}
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="password">Password</Label>
                    <Input
                      id="password"
                      type="password"
                      disabled={isSubmitting}
                      {...register('password')}
                    />
                    {errors.password && (
                      <p className="text-sm text-destructive">{errors.password.message}</p>
                    )}
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="confirmPassword">Confirm Password</Label>
                    <Input
                      id="confirmPassword"
                      type="password"
                      disabled={isSubmitting}
                      {...register('confirmPassword')}
                    />
                    {errors.confirmPassword && (
                      <p className="text-sm text-destructive">{errors.confirmPassword.message}</p>
                    )}
                  </div>
                  <Button disabled={isSubmitting} className="w-full">
                    {isSubmitting && (
                      <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    )}
                    Create Account
                  </Button>
                </div>
              </form>
            </CardContent>
            <CardFooter>
              <div className="text-sm text-muted-foreground text-center w-full">
                Already have an account?{' '}
                <Link href="/login" className="text-primary underline-offset-4 hover:underline">
                  Sign in
                </Link>
              </div>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
}
