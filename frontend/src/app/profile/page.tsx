"use client";

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";
import Link from "next/link";

const profileSchema = z.object({
  username: z.string().min(2, "Username must be at least 2 characters").optional(),
  email: z.string().email("Invalid email"),
});

const passwordSchema = z.object({
  currentPassword: z.string().min(1, "Current password is required"),
  newPassword: z.string().min(6, "New password must be at least 6 characters"),
  confirmPassword: z.string().min(1, "Please confirm your password"),
}).refine((data) => data.newPassword === data.confirmPassword, {
  message: "Passwords do not match",
  path: ["confirmPassword"],
});

type ProfileFormData = z.infer<typeof profileSchema>;
type PasswordFormData = z.infer<typeof passwordSchema>;

export default function ProfilePage() {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"profile" | "password" | "picture">("profile");
  const [uploading, setUploading] = useState(false);

  const profileForm = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
  });

  const passwordForm = useForm<PasswordFormData>({
    resolver: zodResolver(passwordSchema),
  });

  useEffect(() => {
    loadUserProfile();
  }, []);

  const loadUserProfile = async () => {
    try {
      const userData = localStorage.getItem("user");
      if (!userData) {
        window.location.href = "/login";
        return;
      }

      const userInfo = JSON.parse(userData);
      const response = await fetch(`http://localhost:5000/api/profile/${userInfo.id}`);
      
      if (response.ok) {
        const profileData = await response.json();
        setUser(profileData);
        profileForm.reset({
          username: profileData.username || "",
          email: profileData.email,
        });
      } else {
        toast.error("Failed to load profile");
      }
    } catch (error) {
      toast.error("Error loading profile");
    } finally {
      setLoading(false);
    }
  };

  const updateProfile = async (data: ProfileFormData) => {
    try {
      const response = await fetch(`http://localhost:5000/api/profile/${user.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (response.ok) {
        toast.success("Profile updated successfully");
        setUser(result.user);
        // Update localStorage with new user data and notify components
        localStorage.setItem("user", JSON.stringify(result.user));
        window.dispatchEvent(new Event('profileUpdated'));
      } else {
        toast.error(result.error || "Failed to update profile");
      }
    } catch (error) {
      toast.error("Error updating profile");
    }
  };

  const changePassword = async (data: PasswordFormData) => {
    try {
      const response = await fetch(`http://localhost:5000/api/profile/${user.id}/password`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_password: data.currentPassword,
          new_password: data.newPassword,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        toast.success("Password changed successfully");
        passwordForm.reset();
      } else {
        toast.error(result.error || "Failed to change password");
      }
    } catch (error) {
      toast.error("Error changing password");
    }
  };

  const handlePictureUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/profile/${user.id}/picture`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        toast.success("Profile picture updated successfully");
        const updatedUser = result.user || { ...user, profile_picture: result.profile_picture };
        setUser(updatedUser);
        
        // Update localStorage and notify other components
        localStorage.setItem("user", JSON.stringify(updatedUser));
        window.dispatchEvent(new Event('profileUpdated'));
      } else {
        toast.error(result.error || "Failed to upload picture");
      }
    } catch (error) {
      toast.error("Error uploading picture");
    } finally {
      setUploading(false);
    }
  };

  const deletePicture = async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/profile/${user.id}/picture`, {
        method: "DELETE",
      });

      const result = await response.json();

      if (response.ok) {
        toast.success("Profile picture deleted successfully");
        setUser({ ...user, profile_picture: null });
        
        // Update localStorage and notify other components
        const updatedUser = { ...user, profile_picture: null };
        localStorage.setItem("user", JSON.stringify(updatedUser));
        window.dispatchEvent(new Event('profileUpdated'));
      } else {
        toast.error(result.error || "Failed to delete picture");
      }
    } catch (error) {
      toast.error("Error deleting picture");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Unified Layout Container */}
      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Header Section */}
        <div className="mb-10">
          <Link 
            href="/dashboard" 
            className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium mb-4 inline-flex items-center transition-colors"
          >
            ← Back to Dashboard
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-3 tracking-tight">
            Profile Settings
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Manage your account settings and preferences
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200 dark:border-gray-700 mb-8">
          <nav className="-mb-px flex space-x-12">
            {[
              { key: "profile", label: "Profile Info" },
              { key: "password", label: "Password" },
              { key: "picture", label: "Profile Picture" },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`py-3 px-1 border-b-2 font-semibold text-base transition-all duration-200 cursor-pointer ${
                  activeTab === tab.key
                    ? "border-blue-500 text-blue-600 dark:text-blue-400"
                    : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="space-y-8">
          {activeTab === "profile" && (
            <Card className="p-8 shadow-md rounded-xl border-0 bg-white dark:bg-gray-800">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Profile Information</h2>
              <form onSubmit={profileForm.handleSubmit(updateProfile)} className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Username
                  </label>
                  <input
                    type="text"
                    placeholder="Enter your username"
                    className="w-full px-4 py-3 text-base border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    {...profileForm.register("username")}
                  />
                  {profileForm.formState.errors.username && (
                    <p className="text-sm text-red-500 mt-2">
                      {profileForm.formState.errors.username.message}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Email
                  </label>
                  <input
                    type="email"
                    className="w-full px-4 py-3 text-base border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    {...profileForm.register("email")}
                  />
                  {profileForm.formState.errors.email && (
                    <p className="text-sm text-red-500 mt-2">
                      {profileForm.formState.errors.email.message}
                    </p>
                  )}
                </div>

                <div className="pt-4">
                  <Button 
                    type="submit" 
                    className="px-8 py-3 bg-blue-600 hover:bg-blue-700 active:scale-95 transition-all duration-200 font-semibold"
                  >
                    Update Profile
                  </Button>
                </div>
              </form>
            </Card>
          )}

          {activeTab === "password" && (
            <Card className="p-8 shadow-md rounded-xl border-0 bg-white dark:bg-gray-800">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Change Password</h2>
              <form onSubmit={passwordForm.handleSubmit(changePassword)} className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Current Password
                  </label>
                  <input
                    type="password"
                    className="w-full px-4 py-3 text-base border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    {...passwordForm.register("currentPassword")}
                  />
                  {passwordForm.formState.errors.currentPassword && (
                    <p className="text-sm text-red-500 mt-2">
                      {passwordForm.formState.errors.currentPassword.message}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    New Password
                  </label>
                  <input
                    type="password"
                    className="w-full px-4 py-3 text-base border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    {...passwordForm.register("newPassword")}
                  />
                  {passwordForm.formState.errors.newPassword && (
                    <p className="text-sm text-red-500 mt-2">
                      {passwordForm.formState.errors.newPassword.message}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    className="w-full px-4 py-3 text-base border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    {...passwordForm.register("confirmPassword")}
                  />
                  {passwordForm.formState.errors.confirmPassword && (
                    <p className="text-sm text-red-500 mt-2">
                      {passwordForm.formState.errors.confirmPassword.message}
                    </p>
                  )}
                </div>

                <div className="pt-4">
                  <Button 
                    type="submit" 
                    className="px-8 py-3 bg-blue-600 hover:bg-blue-700 active:scale-95 transition-all duration-200 font-semibold"
                  >
                    Change Password
                  </Button>
                </div>
              </form>
            </Card>
          )}

          {activeTab === "picture" && (
            <Card className="p-8 shadow-md rounded-xl border-0 bg-white dark:bg-gray-800">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Profile Picture</h2>
              <div className="space-y-6">
                {/* Current Picture Display */}
                <div className="flex items-center space-x-6">
                  {user?.profile_picture ? (
                    <img
                      src={`http://localhost:5000${user.profile_picture}`}
                      alt="Profile"
                      className="w-24 h-24 rounded-full object-cover border-4 border-gray-200 dark:border-gray-600 shadow-lg"
                    />
                  ) : (
                    <div className="w-24 h-24 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg">
                      <span className="text-white text-2xl font-bold">
                        {user?.username?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || "U"}
                      </span>
                    </div>
                  )}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                      {user?.profile_picture ? "Current profile picture" : "No profile picture set"}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Upload a new picture to personalize your profile
                    </p>
                  </div>
                </div>

                {/* Upload Section */}
                <div className="space-y-3">
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Upload New Picture
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handlePictureUpload}
                    disabled={uploading}
                    className="block w-full text-base text-gray-700 dark:text-gray-300 file:mr-6 file:py-3 file:px-6 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 file:transition-all file:duration-200 border border-gray-300 dark:border-gray-600 rounded-lg"
                  />
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Supported formats: PNG, JPG, JPEG, GIF. Max size: 5MB
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-4 pt-4">
                  {user?.profile_picture && (
                    <Button
                      variant="destructive"
                      onClick={deletePicture}
                      className="px-6 py-3 font-semibold active:scale-95 transition-all duration-200"
                    >
                      Remove Picture
                    </Button>
                  )}
                </div>

                {uploading && (
                  <div className="text-center py-6">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-3"></div>
                    <p className="text-base text-gray-600 dark:text-gray-400 font-medium">Uploading...</p>
                  </div>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
