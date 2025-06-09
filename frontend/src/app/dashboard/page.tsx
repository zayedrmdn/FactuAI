"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function DashboardPage() {
  const router = useRouter();
  const [checkingAuth, setCheckingAuth] = useState(true);

  useEffect(() => {
    const user = localStorage.getItem("user");
    if (!user) {
      router.push("/login");
    } else {
      setCheckingAuth(false);
    }
  }, [router]);

  if (checkingAuth) {
    return <p className="text-center mt-10">Checking authentication...</p>;
  }

  return (
    <main className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="text-3xl font-semibold mb-4">Welcome to your dashboard</h1>
        <button
          onClick={() => {
            localStorage.removeItem("user");
            router.push("/login");
          }}
          className="px-4 py-2 bg-red-500 text-white rounded"
        >
          Log out
        </button>
      </div>
    </main>
  );
}
