'use client';

import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/button';
import { Input, Label } from '@/components/ui/form-controls';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from 'sonner';
import { Loader2, Camera, Trash2 } from 'lucide-react';
import UserAvatar from '@/components/UserAvatar';
import { useUser } from '@/lib/hooks/useUser';

const profileSchema = z.object({
  username: z.string().min(2, 'Username must be at least 2 characters').optional(),
  email: z.string().email('Invalid email'),
});

const passwordSchema = z
  .object({
    currentPassword: z.string().min(1, 'Current password is required'),
    newPassword: z.string().min(6, 'New password must be at least 6 characters'),
    confirmPassword: z.string().min(1, 'Please confirm your password'),
  })
  .refine((data) => data.newPassword === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type ProfileFormData = z.infer<typeof profileSchema>;
type PasswordFormData = z.infer<typeof passwordSchema>;

export default function ProfilePage() {
  const { user, loading: userLoading, refetch } = useUser();
  const [uploading, setUploading] = useState(false);

  const profileForm = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
  });

  const passwordForm = useForm<PasswordFormData>({
    resolver: zodResolver(passwordSchema),
  });

  useEffect(() => {
    if (user) {
      profileForm.reset({
        username: user.username || '',
        email: user.email,
      });
    }
  }, [user, profileForm]);

  const updateProfile = async (data: ProfileFormData) => {
    try {
      if (!user) {
        toast.error('Not logged in');
        return;
      }
      const userId = user.id;
      const response = await fetch(`/api/profile/${userId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (response.ok) {
        toast.success('Profile updated successfully');
        // Refetch user data to update the global state
        refetch();
        globalThis.dispatchEvent(new Event('profileUpdated'));
      } else {
        toast.error(result.error || 'Failed to update profile');
      }
    } catch (err) {
      console.error('Profile update error:', err);
      toast.error('Error updating profile');
    }
  };

  const changePassword = async (data: PasswordFormData) => {
    try {
      if (!user) {
        toast.error('Not logged in');
        return;
      }
      const userId = user.id;
      const response = await fetch(`/api/profile/${userId}/password`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_password: data.currentPassword,
          new_password: data.newPassword,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        toast.success('Password changed successfully');
        passwordForm.reset();
      } else {
        toast.error(result.error || 'Failed to change password');
      }
    } catch (err) {
      console.error('Password change error:', err);
      toast.error('Error changing password');
    }
  };

  const handlePictureUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!user) {
      toast.error('Not logged in');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    try {
      const userId = user.id;
      const response = await fetch(`/api/profile/${userId}/picture`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        toast.success('Profile picture updated successfully');
        // Refetch user data to update the global state
        refetch();
        globalThis.dispatchEvent(new Event('profileUpdated'));
      } else {
        toast.error(result.error || 'Failed to upload picture');
      }
    } catch (err) {
      console.error('Picture upload error:', err);
      toast.error('Error uploading picture');
    } finally {
      setUploading(false);
    }
  };

  const deletePicture = async () => {
    try {
      if (!user) {
        toast.error('Not logged in');
        return;
      }
      const userId = user.id;
      const response = await fetch(`/api/profile/${userId}/picture`, {
        method: 'DELETE',
      });

      const result = await response.json();

      if (response.ok) {
        toast.success('Profile picture deleted successfully');
        // Refetch user data to update the global state
        refetch();
        globalThis.dispatchEvent(new Event('profileUpdated'));
      } else {
        toast.error(result.error || 'Failed to delete picture');
      }
    } catch (err) {
      console.error('Picture delete error:', err);
      toast.error('Error deleting picture');
    }
  };

  if (userLoading || !user) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Profile Settings</h3>
        <p className="text-sm text-muted-foreground">
          Manage your account settings and preferences.
        </p>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Personal Information</CardTitle>
              <CardDescription>Update your personal details here.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-col items-center gap-4 sm:flex-row">
                <UserAvatar />
                <div className="flex flex-col gap-2">
                  <Label htmlFor="picture" className="cursor-pointer">
                    <div className="flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm hover:bg-accent hover:text-accent-foreground">
                      <Camera className="h-4 w-4" />
                      {uploading ? 'Uploading...' : 'Change Picture'}
                    </div>
                    <input
                      id="picture"
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handlePictureUpload}
                      disabled={uploading}
                    />
                  </Label>
                  {user?.profile_picture && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={deletePicture}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="mr-2 h-4 w-4" />
                      Remove
                    </Button>
                  )}
                </div>
              </div>

              <form onSubmit={profileForm.handleSubmit(updateProfile)} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input id="username" {...profileForm.register('username')} />
                  {profileForm.formState.errors.username && (
                    <p className="text-sm text-destructive">
                      {profileForm.formState.errors.username.message}
                    </p>
                  )}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" {...profileForm.register('email')} />
                  {profileForm.formState.errors.email && (
                    <p className="text-sm text-destructive">
                      {profileForm.formState.errors.email.message}
                    </p>
                  )}
                </div>
                <Button type="submit" disabled={profileForm.formState.isSubmitting}>
                  {profileForm.formState.isSubmitting && (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Save Changes
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Security</CardTitle>
              <CardDescription>Manage your password and security settings.</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={passwordForm.handleSubmit(changePassword)} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="currentPassword">Current Password</Label>
                  <Input
                    id="currentPassword"
                    type="password"
                    {...passwordForm.register('currentPassword')}
                  />
                  {passwordForm.formState.errors.currentPassword && (
                    <p className="text-sm text-destructive">
                      {passwordForm.formState.errors.currentPassword.message}
                    </p>
                  )}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="newPassword">New Password</Label>
                  <Input
                    id="newPassword"
                    type="password"
                    {...passwordForm.register('newPassword')}
                  />
                  {passwordForm.formState.errors.newPassword && (
                    <p className="text-sm text-destructive">
                      {passwordForm.formState.errors.newPassword.message}
                    </p>
                  )}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm New Password</Label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    {...passwordForm.register('confirmPassword')}
                  />
                  {passwordForm.formState.errors.confirmPassword && (
                    <p className="text-sm text-destructive">
                      {passwordForm.formState.errors.confirmPassword.message}
                    </p>
                  )}
                </div>
                <Button type="submit" disabled={passwordForm.formState.isSubmitting}>
                  {passwordForm.formState.isSubmitting && (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Update Password
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
