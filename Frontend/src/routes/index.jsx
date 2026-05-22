import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Activity, Mail, Lock, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const Route = createFileRoute("/")({
  component: AuthPage,
});

function AuthPage() {
  const [mode, setMode] = useState("signin");
  const navigate = useNavigate();

  const submit = (e) => {
    e.preventDefault();
    navigate({ to: "/dashboard" });
  };

  return (
    <div className="min-h-screen w-full grid place-items-center px-4 bg-[radial-gradient(ellipse_at_top,_oklch(0.28_0.05_155/_0.25),transparent_60%)]">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-center gap-2 mb-8">
          <div className="h-10 w-10 rounded-xl bg-primary text-primary-foreground grid place-items-center">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="text-lg font-semibold tracking-tight">SuhasVision</div>
            <div className="text-xs text-muted-foreground">Pro cricket biomechanics</div>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card p-1 mb-6">
          <div className="grid grid-cols-2 gap-1">
            {["signin", "signup"].map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={
                  "py-2.5 text-sm rounded-xl transition-colors " +
                  (mode === m
                    ? "bg-primary text-primary-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                {m === "signin" ? "Sign In" : "Create Account"}
              </button>
            ))}
          </div>
        </div>

        <form
          onSubmit={submit}
          className="rounded-2xl border border-border bg-card p-6 space-y-4 shadow-2xl shadow-black/40"
        >
          <h1 className="text-xl font-semibold tracking-tight">
            {mode === "signin" ? "Welcome back, Coach" : "Set up your academy"}
          </h1>
          <p className="text-sm text-muted-foreground -mt-2">
            {mode === "signin"
              ? "Login to continue analyzing your players."
              : "Create your account to onboard your roster."}
          </p>

          {mode === "signup" && (
            <Field icon={User} label="Full name" id="name" placeholder="Rahul Dravid" />
          )}
          <Field icon={Mail} label="Email" id="email" type="email" placeholder="coach@academy.in" />
          <Field icon={Lock} label="Password" id="password" type="password" placeholder="••••••••" />
          {mode === "signup" && (
            <Field icon={Lock} label="Confirm password" id="confirm" type="password" placeholder="••••••••" />
          )}

          <Button type="submit" className="w-full h-11 text-sm font-semibold">
            {mode === "signin" ? "Login" : "Register"}
          </Button>

          <div className="relative py-1">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-[11px] uppercase tracking-wider">
              <span className="bg-card px-2 text-muted-foreground">or</span>
            </div>
          </div>

          <Button
            type="button"
            variant="outline"
            className="w-full h-11 text-sm font-medium"
            onClick={() => navigate({ to: "/dashboard" })}
          >
            <GoogleMark /> Continue with Google
          </Button>
        </form>

        <p className="text-center text-xs text-muted-foreground mt-6">
          Built for academies, coaches and serious batters.
        </p>
      </div>
    </div>
  );
}

function Field({ icon: Icon, label, id, type = "text", placeholder }) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-xs font-medium text-muted-foreground">
        {label}
      </Label>
      <div className="relative">
        <Icon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input id={id} type={type} placeholder={placeholder} className="pl-9 h-11 bg-input/40" />
      </div>
    </div>
  );
}

function GoogleMark() {
  return (
    <svg className="h-4 w-4 mr-2" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#EA4335" d="M12 10.2v3.9h5.5c-.24 1.4-1.7 4.1-5.5 4.1-3.3 0-6-2.7-6-6.1S8.7 6 12 6c1.9 0 3.1.8 3.8 1.5l2.6-2.5C16.8 3.5 14.6 2.5 12 2.5 6.8 2.5 2.5 6.8 2.5 12S6.8 21.5 12 21.5c6.9 0 9.5-4.8 9.5-9.6 0-.6-.1-1.1-.2-1.7H12z"/>
    </svg>
  );
}
