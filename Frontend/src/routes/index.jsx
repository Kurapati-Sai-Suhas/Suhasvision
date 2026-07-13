import React, { useMemo, useState } from "react";
import { useNavigate, createFileRoute } from "@tanstack/react-router";
import {
  ArrowRight,
  ChevronRight,
  Eye,
  EyeOff,
  GraduationCap,
  Mail,
  Lock,
  ShieldCheck,
  Sparkles,
  User,
  Users,
} from "lucide-react";
// import { useAuth } from "@/context/AuthContext";
import { BrandMark } from "@/components/AppShell";
import { toast } from "sonner";

const HERO_IMAGE =
  "https://images.unsplash.com/photo-1750716413756-b66624b64ce4?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjY2NzN8MHwxfHNlYXJjaHwxfHxjcmlja2V0JTIwc3RhZGl1bSUyMGxpZ2h0cyUyMG5pZ2h0fGVufDB8fHx8MTc4MzMxNzg1NXww&ixlib=rb-4.1.0&q=85";

const ROLE_STATS = {
  coach: [
    { label: "Videos reviewed", value: "12,480" },
    { label: "Active academies", value: "38" },
    { label: "Talent invites sent", value: "612" },
  ],
  learner: [
    { label: "Sessions logged", value: "94,210" },
    { label: "Avg score uplift", value: "+21.4%" },
    { label: "Badges unlocked", value: "48,900" },
  ],
};

const AuthPage = () => {
  const navigate = useNavigate();
  

  const [role, setRole] = useState("coach");
  const [mode, setMode] = useState("signin"); // signin | signup
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);

  const stats = useMemo(() => ROLE_STATS[role], [role]);

  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const endpoint = mode === "signin" ? "/api/auth/login/" : "/api/auth/register/";
      
      const payload = mode === "signin" 
        ? { username: email, password } 
        : { email, password, name, role: role.toUpperCase() };

      const apiBase = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiBase}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || data.error || "Authentication failed");
      }

      const userData = data.user || {};
      localStorage.setItem("accessToken", data.access);
      localStorage.setItem("refreshToken", data.refresh);
      localStorage.setItem("user_role", userData.role || role.toUpperCase());
      localStorage.setItem("user_name", userData.name || name || "User");
      
      toast.success(`Welcome to SuhasVision, ${userData.name || "Coach"}`);
      navigate({ to: role === "coach" ? "/coach" : "/learner", replace: true });
    } catch (err) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[#05080F] text-slate-100">
      {/* Ambient background */}
      <div className="pointer-events-none absolute inset-0 sv-ambient" aria-hidden />
      <div className="pointer-events-none absolute inset-0 sv-grid opacity-60" aria-hidden />

      {/* Top nav */}
      <header className="relative z-20 mx-auto flex w-full max-w-[1400px] items-center justify-between px-6 py-6 md:px-10">
        <BrandMark />
        <div className="hidden items-center gap-6 text-sm text-slate-400 md:flex">
          <a data-testid="link-platform" className="sv-link">
            Platform
          </a>
          <a data-testid="link-academies" className="sv-link">
            Academies
          </a>
          <a data-testid="link-research" className="sv-link">
            Research
          </a>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/5 px-3 py-1.5 text-xs text-emerald-200 md:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 sv-pulse" />
            Season 2026 · Live
          </div>
        </div>
      </header>

      <div className="relative z-10 mx-auto grid w-full max-w-[1400px] gap-10 px-6 pb-16 md:px-10 lg:grid-cols-[1.05fr,0.95fr] lg:gap-16 lg:pb-24">
        {/* Left — Hero / narrative */}
        <section className="sv-rise flex flex-col justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-300">
              <Sparkles className="h-3.5 w-3.5 text-emerald-300" />
              <span className="font-medium tracking-[0.14em] uppercase text-slate-400">
                Cricket biomechanics · AI
              </span>
            </div>
            <h1 className="mt-6 font-display text-4xl font-bold leading-[1.05] tracking-tight text-slate-50 sm:text-5xl lg:text-6xl">
              Read every ball,{" "}
              <span className="bg-gradient-to-r from-emerald-300 via-emerald-400 to-[#f5d982] bg-clip-text text-transparent">
                see every micro-movement.
              </span>
            </h1>
            <p className="mt-6 max-w-xl text-base leading-relaxed text-slate-400 sm:text-lg">
              SuhasVision decodes the biomechanics of your batting — balance, power and technique —
              frame by frame. A private studio for elite academies and their next generation.
            </p>

            <div className="mt-8 grid grid-cols-3 gap-4 sm:max-w-lg">
              {stats.map((s) => (
                <div
                  key={s.label}
                  className="rounded-2xl border border-white/5 bg-white/[0.02] p-4"
                  data-testid={`hero-stat-${s.label.replace(/\s+/g, "-").toLowerCase()}`}
                >
                  <div className="font-display text-2xl font-bold tracking-tight text-slate-50">
                    {s.value}
                  </div>
                  <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Testimonial-style card */}
          <div className="relative mt-10 hidden overflow-hidden rounded-3xl border border-white/5 sv-glass lg:block">
            <img
              src={HERO_IMAGE}
              alt="Cricket stadium at night"
              className="absolute inset-0 h-full w-full object-cover opacity-45"
            />
            <div className="absolute inset-0 bg-gradient-to-tr from-[#05080F] via-[#05080F]/70 to-transparent" />
            <div className="relative flex flex-col gap-3 p-8">
              <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-emerald-300">
                Coach's log
              </div>
              <p className="font-display text-xl font-semibold leading-snug text-slate-50">
                "Three of my top-order batters crossed a peak score of 90 in one season. The
                biomechanical overlay changed how we practice."
              </p>
              <div className="mt-3 flex items-center gap-3">
                <div className="h-10 w-10 overflow-hidden rounded-full border border-white/10">
                  <img
                    alt="Suhas Menon"
                    src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?crop=faces&fit=crop&w=200&h=200"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div>
                  <div className="text-sm font-semibold text-slate-100">Suhas Menon</div>
                  <div className="text-xs text-slate-400">Head Batting Coach · Deccan Cricket Academy</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Right — Auth card */}
        <section className="sv-rise sv-rise-2 flex items-center">
          <div className="relative w-full overflow-hidden rounded-3xl border border-white/5 sv-glass p-6 md:p-8">
            <div className="pointer-events-none absolute -top-24 -right-24 h-64 w-64 rounded-full bg-emerald-500/10 blur-3xl" />
            <div className="pointer-events-none absolute -bottom-24 -left-24 h-64 w-64 rounded-full bg-[#D4AF37]/10 blur-3xl" />

            {/* Role toggle */}
            <div
              data-testid="role-toggle"
              className="relative flex rounded-full border border-white/10 bg-white/[0.03] p-1"
            >
              <button
                data-testid="role-coach"
                onClick={() => setRole("coach")}
                className={`relative z-10 flex flex-1 items-center justify-center gap-2 rounded-full px-4 py-2.5 text-sm font-medium transition-colors duration-300 ${
                  role === "coach" ? "text-emerald-950" : "text-slate-300 hover:text-slate-100"
                }`}
              >
                <Users className="h-4 w-4" />
                Sign in as Coach
              </button>
              <button
                data-testid="role-learner"
                onClick={() => setRole("learner")}
                className={`relative z-10 flex flex-1 items-center justify-center gap-2 rounded-full px-4 py-2.5 text-sm font-medium transition-colors duration-300 ${
                  role === "learner" ? "text-emerald-950" : "text-slate-300 hover:text-slate-100"
                }`}
              >
                <GraduationCap className="h-4 w-4" />
                Sign in as Learner
              </button>
              <div
                className={`absolute top-1 bottom-1 left-1 w-[calc(50%-4px)] rounded-full bg-gradient-to-r from-emerald-300 to-emerald-500 shadow-[0_10px_30px_-8px_rgba(16,185,129,0.6)] transition-transform duration-500 ease-out ${
                  role === "learner" ? "translate-x-[calc(100%+0px)]" : "translate-x-0"
                }`}
              />
            </div>

            <div className="mt-8 flex items-end justify-between">
              <div>
                <h2 className="font-display text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
                  {mode === "signin" ? "Welcome back" : "Create your account"}
                </h2>
                <p className="mt-1.5 text-sm text-slate-400">
                  {role === "coach"
                    ? "Access your Coach Console, review queue and talent pipeline."
                    : "Log practice sessions, unlock badges, and track your biomechanics."}
                </p>
              </div>
              <div className="hidden items-center gap-1.5 rounded-full border border-white/5 bg-white/[0.02] px-2.5 py-1.5 text-[10px] uppercase tracking-[0.16em] text-slate-400 sm:flex">
                <ShieldCheck className="h-3 w-3 text-emerald-300" />
                Secure
              </div>
            </div>

            <form onSubmit={handleSubmit} className="mt-8 flex flex-col gap-4">
              {mode === "signup" ? (
                <label className="block">
                  <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                    Full name
                  </span>
                  <div className="group relative">
                    <User className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-emerald-300" />
                    <input
                      data-testid="input-name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="e.g. Arya Patel"
                      className="h-12 w-full rounded-xl border border-white/10 bg-[#05080F] pl-10 pr-4 text-sm text-slate-100 outline-none transition-all placeholder:text-slate-600 focus:border-emerald-400/60 focus:ring-1 focus:ring-emerald-400/40"
                    />
                  </div>
                </label>
              ) : null}

              <label className="block">
                <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Email
                </span>
                <div className="group relative">
                  <Mail className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-emerald-300" />
                  <input
                    data-testid="input-email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder={role === "coach" ? "suhas@academy.cricket" : "arya@academy.cricket"}
                    className="h-12 w-full rounded-xl border border-white/10 bg-[#05080F] pl-10 pr-4 text-sm text-slate-100 outline-none transition-all placeholder:text-slate-600 focus:border-emerald-400/60 focus:ring-1 focus:ring-emerald-400/40"
                  />
                </div>
              </label>

              <label className="block">
                <span className="mb-1.5 flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  <span>Password</span>
                  {mode === "signin" ? (
                    <button
                      type="button"
                      data-testid="link-forgot"
                      className="text-[10px] tracking-normal text-slate-400 normal-case hover:text-emerald-300"
                    >
                      Forgot password?
                    </button>
                  ) : null}
                </span>
                <div className="group relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-emerald-300" />
                  <input
                    data-testid="input-password"
                    type={showPass ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="h-12 w-full rounded-xl border border-white/10 bg-[#05080F] pl-10 pr-11 text-sm text-slate-100 outline-none transition-all placeholder:text-slate-600 focus:border-emerald-400/60 focus:ring-1 focus:ring-emerald-400/40"
                  />
                  <button
                    type="button"
                    data-testid="toggle-password"
                    onClick={() => setShowPass((s) => !s)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-slate-500 hover:text-slate-200"
                  >
                    {showPass ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </label>

              <button
                type="submit"
                data-testid="submit-auth"
                disabled={loading}
                className="sv-sheen group mt-2 inline-flex h-12 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-300 via-emerald-400 to-emerald-500 text-sm font-semibold text-emerald-950 shadow-[0_16px_40px_-16px_rgba(16,185,129,0.7)] transition-all duration-300 hover:from-emerald-200 hover:via-emerald-300 hover:to-emerald-400 active:scale-[0.98] disabled:opacity-70"
              >
                {loading
                  ? "Signing in…"
                  : mode === "signin"
                  ? role === "coach"
                    ? "Enter Coach Console"
                    : "Enter Player Workspace"
                  : "Create account"}
                <ArrowRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5" />
              </button>

              <div className="mt-2 flex items-center gap-3 text-xs text-slate-500">
                <span className="h-px flex-1 bg-white/5" />
                <span className="uppercase tracking-[0.2em]">Or</span>
                <span className="h-px flex-1 bg-white/5" />
              </div>

              <button
                type="button"
                data-testid="toggle-mode"
                onClick={() => setMode((m) => (m === "signin" ? "signup" : "signin"))}
                className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3 text-sm text-slate-300 transition-all hover:border-white/20 hover:bg-white/5 hover:text-slate-100"
              >
                {mode === "signin" ? "New here? Create an account" : "Have an account? Sign in"}
                <ChevronRight className="h-4 w-4" />
              </button>
            </form>

            <div className="mt-6 rounded-xl border border-white/5 bg-white/[0.02] p-3 text-[11px] leading-relaxed text-slate-500">
              By continuing you agree to SuhasVision's academy terms & responsible-AI usage policy.
              All biomechanics data stays inside your academy vault.
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default AuthPage;

export const Route = createFileRoute("/")({
  component: AuthPage,
});
