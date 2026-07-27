import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { fetchWithAuth } from "@/lib/api";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  Award,
  BadgeCheck,
  Check,
  ChevronRight,
  CloudUpload,
  Film,
  Flame,
  Info,
  Inbox as InboxIcon,
  Lock,
  MessageCircle,
  Play,
  Shield,
  Sparkles,
  Star,
  Target,
  Trophy,
  Video,
  X,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import AppShell from "@/components/AppShell";



import { createContext, useContext } from 'react';
const LearnerContext = createContext(null);

const useLearner = () => useContext(LearnerContext);

const iconFor = (name) => {
  const map = { trophy: Trophy, shield: Shield, sparkles: Sparkles, flame: Flame, zap: Zap, award: Award };
  const C = map[name] || Trophy;
  return <C className="h-5 w-5" />;
};

const Section = ({ eyebrow, title, subtitle, action, children }) => (
  <section className="sv-rise">
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-emerald-300">
          {eyebrow}
        </div>
        <h2 className="mt-1 font-display text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
          {title}
        </h2>
        {subtitle ? <p className="mt-1.5 max-w-2xl text-sm text-slate-400">{subtitle}</p> : null}
      </div>
      {action}
    </div>
    {children}
  </section>
);

/* ------------------------------ OVERVIEW ------------------------------ */
const OverviewSection = ({ onNavigate }) => {
  const { learner, learnerUploads, badges } = useLearner();
  const pct = Math.round((learner.tierProgress / 100) * 100);
  const scores = learnerUploads.map((u) => u.overall || 0);
  const peakScore = scores.length ? Math.max(...scores) : 0;
  const oneWeekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const sessionsThisWeek = learnerUploads.filter((u) => new Date(u.dateRaw).getTime() >= oneWeekAgo).length;
  return (
    <div className="flex flex-col gap-10">
      <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass p-6 md:p-10">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-emerald-500/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 -left-24 h-72 w-72 rounded-full bg-[#D4AF37]/12 blur-3xl" />
        <div className="relative flex flex-col justify-between gap-8 lg:flex-row lg:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 sv-pulse" />
              {learner.role}
            </div>
            <h1 className="mt-4 font-display text-3xl font-bold tracking-tight text-slate-50 sm:text-4xl lg:text-5xl">
              Ready to raise your game today,{" "}
              <span className="bg-gradient-to-r from-emerald-300 to-[#f5d982] bg-clip-text text-transparent">
                {learner.name.split(" ")[0]}?
              </span>
            </h1>
            <p className="mt-3 max-w-xl text-base text-slate-400">
              {learner.tierProgress}% of the way to {learner.nextTier}. Log today's session to keep your{" "}
              <span className="font-semibold text-[#f5d982]">{learner.streak}-day streak</span> alive.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button
                data-testid="cta-upload"
                onClick={() => onNavigate("upload")}
                className="sv-sheen group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-300 via-emerald-400 to-emerald-500 px-5 py-3 text-sm font-semibold text-emerald-950 shadow-[0_16px_40px_-16px_rgba(16,185,129,0.7)] transition-all hover:from-emerald-200 hover:via-emerald-300 hover:to-emerald-400"
              >
                Upload today's session
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </button>
              <button
                data-testid="cta-progress"
                onClick={() => onNavigate("progress")}
                className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-5 py-3 text-sm font-medium text-slate-200 hover:border-white/20 hover:bg-white/5"
              >
                View progress timeline
              </button>
            </div>
          </div>

          {/* Ring progress card */}
          <div className="relative w-full max-w-xs shrink-0 overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-[#0C1322] to-[#0A1020] p-6 shadow-inner">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-400">
              <span>Season tier</span>
              <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-200">
                {learner.rank ? `#${learner.rank} academy` : "Unranked"}
              </span>
            </div>
            <div className="relative mt-6 flex items-center justify-center">
              <svg width="180" height="180" viewBox="0 0 100 100" className="-rotate-90">
                <circle cx="50" cy="50" r="42" stroke="rgba(255,255,255,0.08)" strokeWidth="6" fill="none" />
                <circle
                  cx="50"
                  cy="50"
                  r="42"
                  stroke="url(#tier-gradient)"
                  strokeWidth="6"
                  strokeLinecap="round"
                  fill="none"
                  strokeDasharray={`${(pct / 100) * 264} 264`}
                />
                <defs>
                  <linearGradient id="tier-gradient" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#10b981" />
                    <stop offset="100%" stopColor="#D4AF37" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                  Current
                </div>
                <div className="font-display text-2xl font-bold text-slate-50">{learner.currentTier}</div>
                <div className="mt-1 text-[10px] text-slate-500">
                  {learner.tierProgress}% → {learner.nextTier}
                </div>
              </div>
            </div>
            <div className="mt-6 grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Streak</div>
                <div className="mt-1 flex items-center gap-1.5 font-display text-lg font-bold text-[#f5d982]">
                  <Flame className="h-4 w-4" />
                  {learner.streak}d
                </div>
              </div>
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Points</div>
                <div className="mt-1 font-display text-lg font-bold text-emerald-300">
                  {learner.totalPoints.toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[
          { label: "Peak AI score", value: peakScore, sub: "personal best", tone: "emerald", testId: "stat-peak" },
          { label: "Total sessions", value: learnerUploads.length, sub: `${sessionsThisWeek} this week`, tone: "emerald", testId: "stat-sessions" },
          { label: "Badges", value: badges.length, sub: badges.length ? "unlocked" : "none yet", tone: "gold", testId: "stat-badges" },
          { label: "Weekly goal", value: `${learner.weeklyProgress}/${learner.weeklyGoal}`, sub: learner.weeklyProgress >= learner.weeklyGoal ? "goal met" : `${learner.weeklyGoal - learner.weeklyProgress} more to go`, tone: "muted", testId: "stat-goal" },
        ].map((s, i) => (
          <div
            key={s.label}
            data-testid={s.testId}
            className={`sv-rise sv-rise-${i + 1} relative overflow-hidden rounded-2xl border border-white/5 sv-glass p-5`}
          >
            <div
              className={`absolute -right-4 -top-4 h-16 w-16 rounded-full blur-2xl ${
                s.tone === "gold" ? "bg-[#D4AF37]/25" : s.tone === "muted" ? "bg-white/5" : "bg-emerald-500/25"
              }`}
            />
            <div className="relative">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                {s.label}
              </div>
              <div className="mt-3 font-display text-3xl font-bold text-slate-50">{s.value}</div>
              <div
                className={`mt-1.5 text-xs ${
                  s.tone === "gold" ? "text-[#f5d982]" : s.tone === "muted" ? "text-slate-400" : "text-emerald-300"
                }`}
              >
                {s.sub}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent uploads mini */}
      <Section
        eyebrow="Latest sessions"
        title="Your recent uploads"
        action={
          <button
            data-testid="see-all-uploads"
            onClick={() => onNavigate("upload")}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.02] px-3 py-1.5 text-xs text-slate-200 hover:border-white/20 hover:bg-white/5"
          >
            See all <ChevronRight className="h-3.5 w-3.5" />
          </button>
        }
      >
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {learnerUploads.map((u, i) => (
            <div
              key={u.id}
              data-testid={`upload-card-${u.id}`}
              onClick={() => onNavigate("feedback")}
              className={`sv-rise sv-rise-${i + 1} group cursor-pointer overflow-hidden rounded-2xl border border-white/5 sv-glass transition-all hover:-translate-y-0.5 hover:border-white/15`}
            >
              <div className="relative aspect-video overflow-hidden">
                {u.thumbnail ? (
                  <img
                    src={u.thumbnail}
                    alt={u.title}
                    className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-105"
                  />
                ) : (
                  <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-white/[0.06] to-transparent">
                    <Film className="h-8 w-8 text-slate-600" />
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-[#05080F] via-transparent to-transparent" />
                <div className="absolute bottom-2.5 left-2.5 flex items-center gap-1.5">
                  {u.duration ? <span className="sv-chip font-mono">{u.duration}</span> : null}
                  <span
                    className={`sv-chip ${
                      u.status === "reviewed" ? "sv-chip-emerald" : "sv-chip-gold"
                    }`}
                  >
                    {u.status}
                  </span>
                </div>
                <button className="absolute bottom-2.5 right-2.5 inline-flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur-md hover:bg-emerald-400 hover:text-emerald-950">
                  <Play className="h-4 w-4" />
                </button>
              </div>
              <div className="flex flex-col gap-3 p-4">
                <div>
                  <div className="truncate text-sm font-medium text-slate-100">{u.title}</div>
                  <div className="mt-0.5 text-xs text-slate-500">{u.date}</div>
                </div>
                <div className="flex items-center justify-between border-t border-white/5 pt-3 text-xs text-slate-400">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-300" />
                      B {u.scores.balance}
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="h-1.5 w-1.5 rounded-full bg-[#f5d982]" />
                      P {u.scores.power}
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="h-1.5 w-1.5 rounded-full bg-sky-300" />
                      T {u.scores.technique}
                    </span>
                  </div>
                  <span className="font-display font-semibold text-slate-100">{u.overall}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
};

/* ------------------------------ UPLOAD CENTER ------------------------------ */
const UploadSection = ({ onNavigate }) => {
  const { learner, refetch } = useLearner();
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadedName, setUploadedName] = useState(null);
  const inputRef = useRef(null);

  const startUpload = async (file) => {
    setUploadedName(file?.name || "batting_session.mp4");
    setUploading(true);
    
    const formData = new FormData();
    formData.append("video", file);
    formData.append("title", file?.name || "Practice Session");
    
    try {
      // Simulate progress for UI feel
      const progressInterval = setInterval(() => {
        setProgress(p => Math.min(p + 15, 90));
      }, 500);

      const res = await fetchWithAuth("/sessions/analyze_stance/", {
        method: "POST",
        body: formData,
      });
      
      clearInterval(progressInterval);
      setProgress(100);
      toast.success("Video analyzed successfully! Your biomechanics report is ready.");
      refetch(); // refresh learner data
      
      setTimeout(() => {
        setUploading(false);
        setProgress(0);
        setUploadedName(null);
        if (onNavigate) onNavigate("overview");
      }, 2000);
    } catch (err) {
      setProgress(0);
      setUploading(false);
      setUploadedName(null);
      toast.error(err.message || "Upload failed");
    }
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) startUpload(file);
  }, []);

  const onChange = (e) => {
    const file = e.target.files?.[0];
    if (file) startUpload(file);
  };

  return (
    <Section
      eyebrow="Upload center"
      title="Drop your latest session"
      subtitle="MP4 or MOV, up to 500MB. Our AI runs balance, power and technique models frame-by-frame within a minute."
    >
      <div className="grid gap-6 lg:grid-cols-[1.4fr,1fr]">
        <div
          data-testid="dropzone"
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`group relative flex min-h-[340px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-3xl border-2 border-dashed p-10 text-center transition-all ${
            dragOver
              ? "border-emerald-400/80 bg-emerald-400/5"
              : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
          }`}
        >
          <div className="pointer-events-none absolute inset-0 sv-grid opacity-40" />
          <input
            ref={inputRef}
            data-testid="file-input"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={onChange}
          />
          <div className="relative flex flex-col items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-emerald-400/25 bg-emerald-400/10 text-emerald-300 shadow-[0_0_40px_-10px_rgba(16,185,129,0.5)] transition-transform duration-500 group-hover:-translate-y-1">
              <CloudUpload className="h-7 w-7" />
            </div>
            <div>
              <div className="font-display text-xl font-semibold text-slate-50">
                Drag & drop your batting video
              </div>
              <div className="mt-1 text-sm text-slate-400">
                or <span className="font-medium text-emerald-300">browse files</span> to upload
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span className="sv-chip">MP4</span>
              <span className="sv-chip">MOV</span>
              <span className="sv-chip">Up to 500MB</span>
            </div>
          </div>

          {uploading || uploadedName ? (
            <div className="relative mt-8 w-full max-w-lg rounded-2xl border border-white/10 bg-[#05080F]/70 p-4 text-left backdrop-blur-xl">
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-300">
                    <Film className="h-4 w-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="truncate text-sm text-slate-100">{uploadedName}</div>
                    <div className="text-xs text-slate-500">
                      {uploading ? "Uploading & analysing…" : "Ready · queued for coach review"}
                    </div>
                  </div>
                </div>
                <div className="font-mono text-xs text-slate-300">{Math.round(progress)}%</div>
              </div>
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-300 to-[#D4AF37] transition-all duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          ) : null}
        </div>

        {/* Guidelines */}
        <div className="flex flex-col gap-4">
          <div className="rounded-3xl border border-white/5 sv-glass p-6">
            <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
              For best AI results
            </div>
            <div className="mt-1 font-display text-lg font-semibold text-slate-50">
              Filming guidelines
            </div>
            <ul className="mt-5 flex flex-col gap-3 text-sm text-slate-400">
              {[
                "Film front-on, facing the batter, waist level",
                "Ensure the full stance and follow-through are in frame",
                "60fps or higher gives cleaner biomechanics traces",
                "Even lighting — avoid heavy shadows on the pitch",
                "10 to 40 second clips per shot type work best",
              ].map((g) => (
                <li key={g} className="flex items-start gap-2">
                  <BadgeCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" />
                  <span>{g}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-3xl border border-white/5 sv-glass p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-300">
                <Info className="h-5 w-5" />
              </div>
              <div>
                <div className="font-display font-semibold text-slate-50">Weekly goal</div>
                <div className="text-xs text-slate-400">
                  {learner.weeklyProgress} of {learner.weeklyGoal} sessions logged
                </div>
              </div>
            </div>
            <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-emerald-300 to-[#D4AF37]"
                style={{
                  width: `${(learner.weeklyProgress / learner.weeklyGoal) * 100}%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
};

/* ------------------------------ PROGRESS TIMELINE ------------------------------ */
const ProgressSection = () => {
  const { progressTimeline, learnerUploads } = useLearner();
  const [range, setRange] = useState("12w");
  const data = progressTimeline;

  // Real quick-stats derived from the actual session history (first vs.
  // latest logged session), not a hardcoded "since W1" snapshot.
  const first = data[0];
  const latest = data[data.length - 1];
  const statFor = (key) => {
    if (!latest) return { value: "—", delta: null };
    const value = latest[key] ?? 0;
    if (!first || data.length < 2) return { value, delta: null };
    return { value, delta: value - (first[key] ?? 0) };
  };
  const quickStats = [
    { label: "Overall", key: "overall", color: "emerald" },
    { label: "Balance", key: "balance", color: "emerald" },
    { label: "Power", key: "power", color: "gold" },
    { label: "Technique", key: "technique", color: "sky" },
  ].map((s) => ({ ...s, ...statFor(s.key) }));

  // Real recent-session timeline, not fabricated badge-unlock history --
  // there's no badge-awarding system behind the scenes yet.
  const milestones = [...learnerUploads]
    .sort((a, b) => new Date(b.dateRaw) - new Date(a.dateRaw))
    .slice(0, 5)
    .map((u) => ({
      date: u.date,
      title: `${u.title} · Overall ${u.overall ?? "—"}`,
    }));

  return (
    <Section
      eyebrow="Progress"
      title="Your 12-week arc"
      subtitle="How your biomechanics scores have moved since you joined the academy. Coach overrides are baked in."
      action={
        <div className="flex items-center gap-2 text-xs">
          {["4w", "8w", "12w"].map((r) => (
            <button
              key={r}
              data-testid={`range-${r}`}
              onClick={() => setRange(r)}
              className={`rounded-full border px-3 py-1.5 uppercase tracking-[0.14em] transition-all ${
                range === r
                  ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200"
                  : "border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/20"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      }
    >
      <div className="rounded-3xl border border-white/5 sv-glass p-6">
        <div className="grid gap-4 border-b border-white/5 pb-6 md:grid-cols-4">
          {quickStats.map((s) => (
            <div key={s.label} data-testid={`progress-${s.label.toLowerCase()}`}>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                {s.label}
              </div>
              <div className="mt-2 flex items-end gap-2">
                <div className="font-display text-3xl font-bold text-slate-50">{s.value}</div>
                {s.delta !== null ? (
                  <div
                    className={`pb-1 text-xs ${
                      s.color === "gold" ? "text-[#f5d982]" : s.color === "sky" ? "text-sky-300" : "text-emerald-300"
                    }`}
                  >
                    {s.delta >= 0 ? "+" : ""}
                    {s.delta} pts
                  </div>
                ) : null}
              </div>
              <div className="mt-2 text-xs text-slate-500">
                {data.length > 0 ? "Since first session" : "No sessions yet"}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ left: -10, right: 10, top: 10, bottom: 0 }}>
              <defs>
                <linearGradient id="pg-overall" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="pg-balance" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="pg-power" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#D4AF37" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#D4AF37" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="pg-technique" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#60a5fa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="week" stroke="#64748b" tickLine={false} axisLine={false} fontSize={11} />
              <YAxis stroke="#64748b" tickLine={false} axisLine={false} fontSize={11} domain={[50, 100]} />
              <Tooltip
                contentStyle={{
                  background: "rgba(12,19,34,0.95)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 12,
                  color: "#f8fafc",
                  fontSize: 12,
                  backdropFilter: "blur(20px)",
                }}
              />
              <Legend
                wrapperStyle={{ color: "#94a3b8", fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase" }}
                iconSize={8}
              />
              <Area type="monotone" name="Overall" dataKey="overall" stroke="#10b981" strokeWidth={3} fill="url(#pg-overall)" />
              <Area type="monotone" name="Balance" dataKey="balance" stroke="#34d399" strokeWidth={1.5} fill="url(#pg-balance)" />
              <Area type="monotone" name="Power" dataKey="power" stroke="#D4AF37" strokeWidth={1.5} fill="url(#pg-power)" />
              <Area type="monotone" name="Technique" dataKey="technique" stroke="#60a5fa" strokeWidth={1.5} fill="url(#pg-technique)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Timeline history */}
      <div className="mt-6 rounded-3xl border border-white/5 sv-glass p-6">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
          Milestones
        </div>
        <div className="mt-1 font-display text-xl font-semibold text-slate-50">Recent sessions</div>
        {milestones.length === 0 ? (
          <div className="mt-6 text-sm text-slate-500">
            No sessions logged yet — this timeline fills in as you upload.
          </div>
        ) : (
          <div className="relative mt-8 pl-6">
            <div className="absolute left-2.5 top-0 h-full w-px bg-gradient-to-b from-emerald-400/60 via-white/10 to-transparent" />
            {milestones.map((m, i) => (
              <div key={i} className="relative mb-6 last:mb-0">
                <span className="absolute -left-[13px] top-1 h-3 w-3 rounded-full border-2 border-[#05080F] bg-emerald-400" />
                <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                  {m.date}
                </div>
                <div className="mt-1 text-sm font-medium text-slate-100">{m.title}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Section>
  );
};

/* ------------------------------ GAMIFICATION ------------------------------ */
const GamificationSection = () => {
  const { learner, badges } = useLearner();
  return (
    <Section
      eyebrow="Player card"
      title="Your gamification profile"
      subtitle="Every session earns you points. Unlock tiers, badges, and academy recognition as you climb."
    >
      <div className="grid gap-6 lg:grid-cols-[1fr,1.4fr]">
        {/* Player card */}
        <div className="relative overflow-hidden rounded-3xl border border-white/10 sv-glass-strong p-6 md:p-8">
          <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-[#D4AF37]/20 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-20 -left-20 h-56 w-56 rounded-full bg-emerald-500/20 blur-3xl" />
          <div className="relative">
            <div className="flex items-center gap-4">
              <div className="h-16 w-16 overflow-hidden rounded-2xl border border-white/15 bg-gradient-to-br from-emerald-300 to-[#D4AF37] p-[2px]">
                <img
                  src={learner.avatar}
                  alt={learner.name}
                  className="h-full w-full rounded-[14px] object-cover"
                />
              </div>
              <div>
                <div className="font-display text-2xl font-bold text-slate-50">
                  {learner.name}
                </div>
                <div className="text-xs text-slate-400">{learner.role}</div>
                <div className="mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-200">
                  <Star className="h-3 w-3" />
                  {learner.currentTier}
                </div>
              </div>
            </div>
            <div className="mt-8 grid grid-cols-3 gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                  <Flame className="h-3 w-3 text-[#f5d982]" />
                  Streak
                </div>
                <div className="mt-2 font-display text-2xl font-bold text-[#f5d982]">
                  {learner.streak}
                  <span className="ml-0.5 text-sm">d</span>
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                  <Sparkles className="h-3 w-3 text-emerald-300" />
                  Points
                </div>
                <div className="mt-2 font-display text-2xl font-bold text-emerald-300">
                  {learner.totalPoints.toLocaleString()}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                  <Trophy className="h-3 w-3 text-emerald-300" />
                  Rank
                </div>
                <div className="mt-2 font-display text-2xl font-bold text-slate-50">
                  {learner.rank ? `#${learner.rank}` : "—"}
                </div>
              </div>
            </div>

            <div className="mt-6">
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                <span>Progress to {learner.nextTier}</span>
                <span className="font-mono text-slate-200">{learner.tierProgress}%</span>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-300 to-[#D4AF37]"
                  style={{ width: `${learner.tierProgress}%` }}
                />
              </div>
            </div>

            <div className="mt-8 flex items-center gap-3 border-t border-white/10 pt-6">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-300">
                <Target className="h-4 w-4" />
              </div>
              <div>
                <div className="text-sm font-medium text-slate-100">Next objective</div>
                <div className="text-xs text-slate-400">
                  {learner.nextTier === "Max tier"
                    ? "You've reached the top tier — keep logging sessions to stay sharp."
                    : `${learner.tierProgress}% of the way to ${learner.nextTier} — every logged session adds points.`}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Badges */}
        <div className="rounded-3xl border border-white/5 sv-glass p-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                Achievements
              </div>
              <div className="mt-1 font-display text-xl font-semibold text-slate-50">
                Badges & unlockables
              </div>
            </div>
            <div className="text-xs text-slate-400">
              {badges.length > 0
                ? `${badges.filter((b) => b.unlocked).length} / ${badges.length} unlocked`
                : "No badges yet"}
            </div>
          </div>
          <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-3">
            {badges.length === 0 ? (
              <div className="col-span-full rounded-2xl border border-white/5 bg-white/[0.01] p-6 text-center text-sm text-slate-500">
                Badges aren't live yet — they'll appear here as the achievement system rolls out.
              </div>
            ) : null}
            {badges.map((b, i) => (
              <div
                key={b.id}
                data-testid={`badge-${b.id}`}
                className={`sv-rise sv-rise-${(i % 6) + 1} relative overflow-hidden rounded-2xl border p-4 transition-all ${
                  b.unlocked
                    ? "border-white/10 bg-white/[0.02] hover:-translate-y-0.5 hover:border-white/20"
                    : "border-white/5 bg-white/[0.01] opacity-70"
                }`}
              >
                <div
                  className={`flex h-11 w-11 items-center justify-center rounded-xl ${
                    b.tone === "gold"
                      ? "bg-gradient-to-br from-[#f5d982] to-[#D4AF37] text-[#3b2c00]"
                      : b.tone === "emerald"
                      ? "bg-gradient-to-br from-emerald-300 to-emerald-500 text-emerald-950"
                      : "bg-white/5 text-slate-500"
                  }`}
                >
                  {b.unlocked ? iconFor(b.icon) : <Lock className="h-4 w-4" />}
                </div>
                <div className="mt-4 text-sm font-semibold text-slate-100">{b.name}</div>
                <div className="mt-1 text-xs text-slate-400">{b.description}</div>
                {b.unlocked ? (
                  <div className="mt-3 inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-emerald-300">
                    <Check className="h-3 w-3" />
                    {b.unlockedAt}
                  </div>
                ) : (
                  <div className="mt-3">
                    <div className="h-1 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-slate-400 to-slate-200"
                        style={{ width: `${b.progress || 0}%` }}
                      />
                    </div>
                    <div className="mt-1.5 text-[10px] font-mono text-slate-500">
                      {b.progress}% complete
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
};

/* ------------------------------ FEEDBACK INBOX ------------------------------ */
const FeedbackSection = () => {
  const { feedbackInbox, learner, markFeedbackSeen } = useLearner();
  const [selectedId, setSelectedId] = useState(feedbackInbox?.[0]?.id);
  const selected = feedbackInbox.find((f) => f.id === selectedId) || feedbackInbox[0];

  // Real read-tracking (persisted to localStorage via markFeedbackSeen in
  // LearnerDashboard) -- opening a feedback item is what actually marks it
  // seen, replacing the previous `unread: s.status === "COMPLETED"` which
  // marked every completed session unread forever regardless of whether
  // the learner had ever looked at it.
  useEffect(() => {
    if (selected?.id != null) markFeedbackSeen(selected.id);
  }, [selected?.id, markFeedbackSeen]);

  return (
    <Section
      eyebrow="Feedback"
      title="Inbox from your coach"
      subtitle="AI analysis for every session, plus any score overrides or notes your coach adds — you'll see it here."
    >
      <div className="grid gap-6 lg:grid-cols-[1fr,1.6fr]">
        <div className="flex flex-col gap-3">
          {feedbackInbox.map((f, i) => (
            <button
              key={f.id}
              data-testid={`feedback-item-${f.id}`}
              onClick={() => setSelectedId(f.id)}
              className={`sv-rise sv-rise-${i + 1} group relative flex flex-col gap-3 rounded-2xl border p-4 text-left transition-all ${
                selectedId === f.id
                  ? "border-emerald-400/40 bg-gradient-to-r from-emerald-500/12 via-emerald-500/4 to-transparent"
                  : "border-white/5 bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {f.unread ? (
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 sv-pulse" />
                  ) : (
                    <span className="h-1.5 w-1.5 rounded-full bg-transparent" />
                  )}
                  <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                    {f.date}
                  </div>
                </div>
                {f.coachOverride ? (
                  <span className="sv-chip sv-chip-gold">Override</span>
                ) : null}
              </div>
              <div className="font-display text-base font-semibold text-slate-100">
                {f.videoTitle}
              </div>
              <div className="line-clamp-2 text-xs text-slate-400">{f.aiSummary}</div>
              <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                <div className="h-6 w-6 overflow-hidden rounded-full border border-white/10">
                  <img src={f.coachAvatar} alt={f.coachName} className="h-full w-full object-cover" />
                </div>
                {f.coachName}
              </div>
            </button>
          ))}
        </div>

        {/* Detail */}
        <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass p-6 md:p-8">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                Video · {selected.date}
              </div>
              <div className="mt-1 font-display text-2xl font-bold text-slate-50">
                {selected.videoTitle}
              </div>
              <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
                <div className="h-7 w-7 overflow-hidden rounded-full border border-white/10">
                  <img src={selected.coachAvatar} alt="coach" className="h-full w-full object-cover" />
                </div>
                Reviewed by {selected.coachName}
              </div>
            </div>
            <button
              data-testid="reply-coach"
              disabled
              title="Direct coach messaging isn't available yet"
              className="inline-flex cursor-not-allowed items-center gap-2 rounded-xl border border-white/5 bg-white/[0.01] px-4 py-2.5 text-sm text-slate-500"
            >
              <MessageCircle className="h-4 w-4" />
              Reply (coming soon)
            </button>
          </div>

          <div className="mt-8 grid gap-5">
            <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-5">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                AI Summary
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-300">{selected.aiSummary}</p>
            </div>

            {/* Real coach note (AnalysisSession.bonus_insight, written by a
                coach's PATCH override -- see coach.jsx). The before/after
                score comparison this card used to show was fabricated
                (previousScore/newScore were never populated from anywhere);
                removed rather than displaying blank values, since the
                per-metric CoachOverrideLog entries this would need aren't
                fetched by this view. */}
            {selected.coachOverride ? (
              <div className="rounded-2xl border border-[#D4AF37]/30 bg-gradient-to-br from-[#D4AF37]/10 to-transparent p-5">
                <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-[#f5d982]">
                  <BadgeCheck className="h-3.5 w-3.5" />
                  Coach note
                </div>
                <p className="mt-3 text-sm leading-relaxed text-slate-300">
                  {selected.coachOverride.note}
                </p>
              </div>
            ) : null}

            {/* SWOT Analysis */}
            <div className="grid grid-cols-2 gap-4">
              {['Strength', 'Weakness', 'Opportunity', 'Threat'].map((type, i) => (
                <div key={type} className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                  <div className={`text-[10px] font-semibold uppercase tracking-[0.24em] ${i === 0 ? 'text-emerald-300' : i === 1 ? 'text-rose-300' : i === 2 ? 'text-sky-300' : 'text-amber-300'}`}>
                    {type}
                  </div>
                  <div className="mt-2 text-sm text-slate-300">{selected.swot?.[i] || ""}</div>
                </div>
              ))}
            </div>

            {/* Weakness Detail -- biomechanical reason only. This card used to
                also claim a specific "struggling length"/"struggling line"
                (e.g. "Short of a Length" / "Body Line"), but that was a
                static 1-of-4 lookup keyed only on which score was lowest --
                never computed from real ball-tracking or line/length data,
                which this system does not capture. Removed rather than
                left in place implying a personalized finding this app
                cannot actually make. */}
            <div className="rounded-2xl border border-rose-500/20 bg-gradient-to-br from-rose-500/5 to-transparent p-5">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-rose-400">
                Weakness Detail
              </div>
              <div className="mt-4">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Biomechanical Reason</div>
                <p className="mt-1 text-sm text-slate-300">{selected.vulnerability?.reason}</p>
              </div>
            </div>

            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                Action Plan & Drills
              </div>
              <ul className="mt-3 flex flex-col gap-3">
                {selected.drills?.map((drill, i) => (
                  <li
                    key={i}
                    className="flex flex-col gap-2 rounded-xl border border-white/5 bg-white/[0.02] p-4 text-sm text-slate-300"
                  >
                    <div className="flex items-center gap-3 font-semibold text-slate-100">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-[11px] text-emerald-300">
                        {i + 1}
                      </span>
                      {drill.name}
                    </div>
                    <div className="pl-9 text-slate-400">{drill.description}</div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
};

// Metric-level fallback used only when a session has no real backend
// attribution to show (older sessions predating the gradient-attribution
// feature, or the reliability-fallback rule scorer, which has no gradients
// to attribute). The backend now runs real per-joint gradient attribution
// (Expected Gradients, computed via ml_service.py's `_expected_gradients`)
// for normal sessions and returns an honest, session-specific weakness
// sentence + concrete drill text in `primary_weakness` / `thing_to_change` —
// see generateAIInsights below, which prefers that real data when present.
const METRIC_LIBRARY = {
  balance: {
    label: "Balance",
    description: "base stability and head position through the shot",
    reasonTemplate: (v) => `Balance (${v}/100) shows instability at the crease — likely falling toward the offside, vulnerable to full, wide deliveries.`,
    drills: [
      { name: "The Flamingo Drill", description: "Play your shot and hold your pose on one leg for 3 seconds to enforce core stability." },
      { name: "Cone Touch Drill", description: "Place a cone outside off. Stride and touch it with your bat while keeping your head perfectly still." },
    ],
  },
  power: {
    label: "Power",
    description: "weight transfer and bat speed through contact",
    reasonTemplate: (v) => `Power (${v}/100) is low — hip rotation looks locked, making it hard to generate force against balls dug in short.`,
    drills: [
      { name: "Medicine Ball Throws", description: "Mimic your batting stance and throw a medicine ball into a wall, focusing on hip rotation." },
      { name: "Heavy Bat Swings", description: "Take 20 shadow swings with a heavier bat to train fast-twitch muscle fibers." },
    ],
  },
  technique: {
    label: "Technique",
    description: "grip, backlift, and bat path correctness",
    reasonTemplate: (v) => `Technique (${v}/100) suggests the bat path comes down at an angle, opening the gate for straight deliveries.`,
    drills: [
      { name: "Wall Drill", description: "Stand close to a wall and practice your straight drive — if your bat hits the wall, the downswing is crooked." },
      { name: "Top Hand Only", description: "Hit balls off a tee using only your top hand to force a straight bat path." },
    ],
  },
  defence: {
    label: "Defence",
    description: "stump coverage and head-over-ball at contact",
    reasonTemplate: (v) => `Defence (${v}/100) is the softer area — stump coverage and head position at contact need tightening against seam movement.`,
    drills: [
      { name: "Leave Decision Drill", description: "Partner throws mixed lines. Practice decisively leaving everything outside the 4th stump." },
      { name: "Soft Hands Drill", description: "Play defensive shots with a deliberately loose grip to reduce the chance of edges carrying." },
    ],
  },
};

const generateAIInsights = (balance, power, technique, defence, session = {}) => {
  const scores = { balance, power, technique, defence };
  const entries = Object.entries(scores);
  const [weakestKey, weakestVal] = entries.reduce((min, e) => (e[1] < min[1] ? e : min));
  const [strongestKey, strongestVal] = entries.reduce((max, e) => (e[1] > max[1] ? e : max));

  const weak = METRIC_LIBRARY[weakestKey];
  const strong = METRIC_LIBRARY[strongestKey];

  // Prefer the backend's real, session-specific weakness sentence (gradient
  // attribution against the actual joints in this video) over the generic
  // metric-level template, when the backend produced one.
  const reason = session.primary_weakness || weak.reasonTemplate(weakestVal);

  // `thing_to_change` carries the real per-joint drill recommendation
  // (semicolon-joined) from gradient attribution. Falls back to the canned
  // metric-level drills only when absent (older sessions, or the
  // reliability-fallback rule scorer, which has no gradients to attribute).
  const realDrills = session.thing_to_change
    ? session.thing_to_change
        .split(";")
        .map((d) => d.trim())
        .filter(Boolean)
        .map((d) => ({ name: "Recommended Drill", description: d }))
    : null;

  const swot = [
    `${strong.label} (${strongestVal}/100) is a genuine strength — ${strong.description}.`,
    `${weak.label} (${weakestVal}/100) needs the most work — ${weak.description}.`,
    `Closing the gap between ${weak.label.toLowerCase()} and the rest of the game is the fastest way to raise the overall score.`,
    reason,
  ];

  return {
    vulnerability: { reason },
    swot,
    drills: realDrills || weak.drills,
  };
};

/* --------------------------------- PAGE --------------------------------- */
const SEEN_FEEDBACK_STORAGE_KEY = "suhasvision.seenFeedbackIds";

const LearnerDashboard = () => {
  const [active, setActive] = useState("overview");

  // Real, working read-tracking for the feedback inbox (localStorage-backed,
  // per-browser). Replaces a prior `unread: s.status === "COMPLETED"` that
  // marked every completed session unread forever -- that was never a
  // reflection of whether the learner had actually opened it.
  const [seenIds, setSeenIds] = useState(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem(SEEN_FEEDBACK_STORAGE_KEY) || "[]"));
    } catch {
      return new Set();
    }
  });
  const markFeedbackSeen = useCallback((id) => {
    setSeenIds((prev) => {
      if (prev.has(id)) return prev;
      const next = new Set(prev);
      next.add(id);
      try {
        localStorage.setItem(SEEN_FEEDBACK_STORAGE_KEY, JSON.stringify([...next]));
      } catch {
        // localStorage unavailable (private browsing, quota) -- read state
        // just won't persist across reloads; not worth failing the UI over.
      }
      return next;
    });
  }, []);

  const titleMap = {
    overview: "My Cricket",
    upload: "Upload Center",
    progress: "Progress Timeline",
    profile: "Gamification Profile",
    feedback: "Feedback Inbox",
  };

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["learnerDashboard"],
    queryFn: () => fetchWithAuth("/learner/me/"),
  });

  if (isLoading || !data) return <div className="flex h-screen items-center justify-center bg-[#05080F] text-emerald-400">Loading your profile...</div>;
  if (error) return <div className="flex h-screen items-center justify-center bg-[#05080F] text-red-400">Error loading profile: {error.message}</div>;

  // Transform backend data to match UI expectations
  const backendProfile = data.profile || {};
  const sessions = data.sessions || [];

  // Tier is a real, deterministic function of the real total_points field
  // (thresholds below), not fabricated. Rank/academy-rank has no honest
  // value to show -- there's no leaderboard endpoint exposing other
  // learners' data yet -- so it renders as "—" rather than a made-up number.
  // Same for a profile photo: no upload feature exists, so this uses a
  // generated initials avatar instead of pretending a stock photo is real.
  const TIER_THRESHOLDS = [
    { name: "Bronze", min: 0 },
    { name: "Silver", min: 500 },
    { name: "Gold", min: 1500 },
    { name: "Platinum", min: 3000 },
  ];
  const points = backendProfile.total_points || 0;
  let tierIdx = 0;
  for (let i = 0; i < TIER_THRESHOLDS.length; i++) {
    if (points >= TIER_THRESHOLDS[i].min) tierIdx = i;
  }
  const currentTierInfo = TIER_THRESHOLDS[tierIdx];
  const nextTierInfo = TIER_THRESHOLDS[tierIdx + 1] || null;
  const tierProgress = nextTierInfo
    ? Math.min(100, Math.max(0, Math.round(((points - currentTierInfo.min) / (nextTierInfo.min - currentTierInfo.min)) * 100)))
    : 100;

  const learnerName = backendProfile.name || "Guest Player";
  const learner = {
    id: backendProfile.id,
    name: learnerName,
    role: `${backendProfile.playing_level || "Batter"} · ${backendProfile.batting_hand || "—"}`,
    email: backendProfile.user?.email || "",
    avatar: `https://ui-avatars.com/api/?name=${encodeURIComponent(learnerName)}&background=10b981&color=05080F&bold=true`,
    streak: backendProfile.current_streak || 0,
    totalPoints: points,
    currentTier: currentTierInfo.name,
    nextTier: nextTierInfo ? nextTierInfo.name : "Max tier",
    tierProgress,
    rank: null, // "—" in the UI; no leaderboard data source exists yet
    weeklyGoal: 3, // fixed app-wide target, not personalized data
    weeklyProgress: sessions.length,
  };

  // No video-duration or thumbnail data exists server-side (the uploaded
  // video is deleted immediately after inference for zero-storage privacy
  // compliance, so there's nothing to generate a real thumbnail from without
  // adding a dedicated feature to capture one at upload time). A neutral
  // generated placeholder is honest about that, instead of a specific stock
  // photo pretending to be a frame from the real video.
  const learnerUploads = sessions.map(s => ({
    id: s.id,
    title: s.title || `Session ${s.id}`,
    date: new Date(s.date_analyzed).toLocaleDateString(),
    dateRaw: s.date_analyzed, // ISO timestamp for real date math (the display `date` above is locale-formatted, not safe to re-parse)
    duration: null,
    thumbnail: null,
    scores: { balance: s.balance_score, power: s.power_score, technique: s.technique_score },
    overall: s.overall_score,
    status: s.status === "COMPLETED" ? "reviewed" : "pending",
  }));

  const progressTimeline = sessions.map((s, i) => ({
    week: `W${i+1}`,
    balance: s.balance_score,
    power: s.power_score,
    technique: s.technique_score,
    overall: s.overall_score
  })).reverse();

  // No badge system exists yet -- an empty array renders an honest "no
  // badges" empty state rather than a placeholder list.
  const badges = [];

  const feedbackInbox = sessions.map((s) => {
    const insights = generateAIInsights(s.balance_score, s.power_score, s.technique_score, s.defence_score, s);
    return {
      id: s.id,
      date: new Date(s.date_analyzed).toLocaleDateString(),
      unread: s.status === "COMPLETED" && !seenIds.has(s.id),
      // Real coach note (AnalysisSession.bonus_insight, written when a
      // coach PATCHes an override -- see coach.jsx's perform_update). No
      // before/after score comparison: those live in per-metric
      // CoachOverrideLog rows this view doesn't fetch, and showing blank
      // values would be worse than not showing them.
      coachOverride: s.bonus_insight ? { note: s.bonus_insight } : false,
      videoTitle: s.title || `Session ${s.id}`,
      aiSummary: `AI Biomechanics Analysis: Balance ${s.balance_score}/100, Power ${s.power_score}/100, Technique ${s.technique_score}/100. Overall AI Score: ${s.overall_score}. ${s.status === "COMPLETED" ? "Analysis complete." : "Processing..."}`,
      // No coach-assignment/profile-photo data is exposed by the API yet --
      // generic label instead of a fabricated name and stock photo.
      coachAvatar: `https://ui-avatars.com/api/?name=Coach&background=1f2937&color=fff`,
      coachName: "Your Coach",
      swot: insights.swot,
      vulnerability: insights.vulnerability,
      drills: insights.drills,
    };
  });

  const contextValue = { learner, learnerUploads, progressTimeline, badges, feedbackInbox, refetch, markFeedbackSeen };

  return (
    <LearnerContext.Provider value={contextValue}>
      <AppShell
        role="learner"
        activeKey={active}
        onNavigate={setActive}
        title={titleMap[active]}
        profile={learner}
        navBadges={{ feedback: feedbackInbox.filter((f) => f.unread).length }}
      >
        {active === "overview" ? <OverviewSection onNavigate={setActive} /> : null}
        {active === "upload" ? <UploadSection onNavigate={setActive} /> : null}
        {active === "progress" ? <ProgressSection /> : null}
        {active === "profile" ? <GamificationSection /> : null}
        {active === "feedback" && feedbackInbox.length > 0 ? <FeedbackSection /> : null}
      </AppShell>
    </LearnerContext.Provider>
  );
};

export default LearnerDashboard;

export const Route = createFileRoute("/learner")({
  component: LearnerDashboard,
});
