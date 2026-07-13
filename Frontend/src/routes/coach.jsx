import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { fetchWithAuth } from "@/lib/api";
import React, { useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  CalendarDays,
  Check,
  ChevronDown,
  Circle,
  Clock,
  Crown,
  Dot,
  Flame,
  Gauge,
  Layers,
  Mail,
  MessageCircle,
  Minus,
  Pause,
  Play,
  Rewind,
  Search,
  Send,
  Sparkles,
  Star,
  Target,
  TrendingUp,
  Trophy,
  UserCheck,
  Video,
  X,
} from "lucide-react";
import { toast } from "sonner";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import AppShell from "@/components/AppShell";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Textarea } from "@/components/ui/textarea";


import { createContext, useContext } from 'react';
const CoachContext = createContext(null);
const useCoach = () => useContext(CoachContext);

const trendIcon = (t) => {
  if (t === "up") return <TrendingUp className="h-3 w-3 text-emerald-300" />;
  if (t === "down") return <TrendingUp className="h-3 w-3 rotate-180 text-rose-300" />;
  return <Minus className="h-3 w-3 text-slate-500" />;
};

const ScoreBar = ({ label, value, tone = "emerald" }) => {
  const bg =
    tone === "gold"
      ? "from-[#f5d982] to-[#D4AF37]"
      : tone === "rose"
      ? "from-rose-300 to-rose-500"
      : "from-emerald-300 to-emerald-500";
  return (
    <div className="flex flex-col gap-1.5" data-testid={`bar-${label.toLowerCase()}`}>
      <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.16em] text-slate-500">
        <span>{label}</span>
        <span className="font-mono text-xs text-slate-200">{value}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${bg}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
};

const Section = ({ eyebrow, title, action, children }) => (
  <section className="sv-rise">
    <div className="mb-5 flex items-end justify-between gap-4">
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-emerald-300">
          {eyebrow}
        </div>
        <h2 className="mt-1 font-display text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
          {title}
        </h2>
      </div>
      {action}
    </div>
    {children}
  </section>
);

/* ------------------------------- OVERVIEW ------------------------------- */
const OverviewSection = ({ onOpenAnalysis }) => {
  const { coach, reviewQueue, coachMetrics, progressTimeline, coachSessions, recentActivity, biomechanicsSnapshot, latestSession } = useCoach();
  return (
    <div className="flex flex-col gap-10">
      {/* Hero header */}
      <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass p-6 md:p-10">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-emerald-500/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 -left-24 h-72 w-72 rounded-full bg-[#D4AF37]/10 blur-3xl" />
        <div className="relative flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 sv-pulse" />
              {coach.name}
            </div>
            <h1 className="mt-4 font-display text-3xl font-bold tracking-tight text-slate-50 sm:text-4xl lg:text-5xl">
              Good morning, coach.
            </h1>
            <p className="mt-3 max-w-xl text-base text-slate-400">
              {reviewQueue.length > 0
                ? `${reviewQueue.length} video${reviewQueue.length === 1 ? "" : "s"} in your review queue.`
                : "No videos waiting in your review queue right now."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              data-testid="cta-open-queue"
              className="sv-sheen group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-300 via-emerald-400 to-emerald-500 px-5 py-3 text-sm font-semibold text-emerald-950 shadow-[0_16px_40px_-16px_rgba(16,185,129,0.7)] transition-all hover:from-emerald-200 hover:via-emerald-300 hover:to-emerald-400"
              onClick={() => onOpenAnalysis(reviewQueue[0])}
            >
              Open review queue
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </button>
            <button
              data-testid="cta-invite"
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-5 py-3 text-sm font-medium text-slate-200 transition-all hover:border-white/20 hover:bg-white/5"
            >
              <Mail className="h-4 w-4" />
              Send new invite
            </button>
          </div>
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {coachMetrics.map((m, i) => (
          <div
            key={m.testId}
            data-testid={m.testId}
            className={`sv-rise sv-rise-${i + 1} group relative overflow-hidden rounded-2xl border border-white/5 sv-glass p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-white/10`}
          >
            <div
              className={`absolute -right-6 -top-6 h-20 w-20 rounded-full blur-2xl transition-opacity duration-500 group-hover:opacity-70 ${
                m.tone === "gold" ? "bg-[#D4AF37]/20" : m.tone === "muted" ? "bg-white/5" : "bg-emerald-500/20"
              }`}
            />
            <div className="relative">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                {m.label}
              </div>
              <div className="mt-3 font-display text-3xl font-bold tracking-tight text-slate-50">
                {m.value}
              </div>
              <div
                className={`mt-1.5 text-xs ${
                  m.tone === "gold"
                    ? "text-[#f5d982]"
                    : m.tone === "muted"
                    ? "text-slate-400"
                    : "text-emerald-300"
                }`}
              >
                {m.delta}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Two-col layout: score chart + today's plan */}
      <div className="grid gap-6 lg:grid-cols-[1.6fr,1fr]">
        <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass p-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                Academy score trend
              </div>
              <div className="mt-1 font-display text-xl font-semibold text-slate-50">
                12-week rolling average
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="sv-chip sv-chip-emerald">Balance</span>
              <span className="sv-chip sv-chip-gold">Power</span>
              <span className="sv-chip">Technique</span>
            </div>
          </div>
          <div className="mt-6 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={progressTimeline} margin={{ left: -10, right: 10, top: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="g-balance" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="g-power" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#D4AF37" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#D4AF37" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="g-technique" x1="0" y1="0" x2="0" y2="1">
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
                <Area type="monotone" dataKey="balance" stroke="#10b981" strokeWidth={2} fill="url(#g-balance)" />
                <Area type="monotone" dataKey="power" stroke="#D4AF37" strokeWidth={2} fill="url(#g-power)" />
                <Area
                  type="monotone"
                  dataKey="technique"
                  stroke="#60a5fa"
                  strokeWidth={2}
                  fill="url(#g-technique)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass p-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                Today
              </div>
              <div className="font-display text-xl font-semibold text-slate-50">Coaching schedule</div>
            </div>
            <CalendarDays className="h-5 w-5 text-slate-500" />
          </div>
          <div className="mt-5 flex flex-col divide-y divide-white/5">
            {coachSessions.length === 0 ? (
              <div className="py-6 text-center text-sm text-slate-500">
                No scheduled sessions — calendar integration coming soon.
              </div>
            ) : null}
            {coachSessions.map((s, i) => (
              <div
                key={i}
                data-testid={`session-${i}`}
                className="group flex items-start justify-between gap-3 py-3.5 transition-colors first:pt-0 last:pb-0 hover:text-slate-100"
              >
                <div className="flex min-w-0 items-start gap-3">
                  <div className="mt-1 font-mono text-xs text-slate-400">{s.time}</div>
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-slate-100">{s.title}</div>
                    <div className="mt-0.5 text-xs text-slate-500">
                      {s.location} · {s.attendees} attending
                    </div>
                  </div>
                </div>
                <ArrowUpRight className="h-4 w-4 shrink-0 text-slate-600 transition-colors group-hover:text-emerald-300" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent activity + biomechanics snapshot */}
      <div className="grid gap-6 lg:grid-cols-[1fr,1.4fr]">
        <div className="rounded-3xl border border-white/5 sv-glass p-6">
          <div className="flex items-center justify-between">
            <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
              Live feed
            </div>
            <span className="text-xs text-slate-500">Last 48 hours</span>
          </div>
          <div className="mt-4 font-display text-xl font-semibold text-slate-50">Academy activity</div>
          <ul className="mt-5 flex flex-col gap-4">
            {recentActivity.map((a) => (
              <li key={a.id} className="flex items-start gap-3">
                <span className="mt-1 flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-emerald-300">
                  {a.icon === "video" ? (
                    <Video className="h-3.5 w-3.5" />
                  ) : a.icon === "star" ? (
                    <Star className="h-3.5 w-3.5" />
                  ) : a.icon === "trend" ? (
                    <TrendingUp className="h-3.5 w-3.5" />
                  ) : (
                    <Mail className="h-3.5 w-3.5" />
                  )}
                </span>
                <div>
                  <div className="text-sm text-slate-200">{a.text}</div>
                  <div className="mt-0.5 text-xs text-slate-500">{a.time}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-3xl border border-white/5 sv-glass p-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                Latest biomechanics snapshot
              </div>
              <div className="mt-1 font-display text-xl font-semibold text-slate-50">
                {latestSession ? `${latestSession.learner} · ${latestSession.title}` : "No sessions yet"}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {latestSession ? `Submitted ${latestSession.date}` : "Waiting on the first upload"}
              </div>
            </div>
            <button
              data-testid="jump-analysis"
              onClick={() => onOpenAnalysis(reviewQueue[0])}
              className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.02] px-3 py-1.5 text-xs text-slate-200 hover:border-white/20 hover:bg-white/5"
            >
              Open in analysis <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="mt-6 grid gap-5 md:grid-cols-3">
            {Object.entries(biomechanicsSnapshot).map(([k, v], i) => (
              <div
                key={k}
                className="rounded-2xl border border-white/5 bg-white/[0.02] p-4"
                data-testid={`snapshot-${k}`}
              >
                <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                  <span>{k}</span>
                  <span className="rounded-full bg-emerald-500/10 px-1.5 py-0.5 text-emerald-300">
                    {v.trend}
                  </span>
                </div>
                <div className="mt-3 font-display text-4xl font-bold text-slate-50">{v.score}</div>
                <div className="mt-3 flex flex-col gap-2">
                  {v.breakdown.map((b) => (
                    <div key={b.label} className="flex items-center justify-between text-xs text-slate-400">
                      <span>{b.label}</span>
                      <span className="font-mono text-slate-200">{b.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

/* ------------------------------ REVIEW QUEUE ------------------------------ */
const ReviewQueueSection = ({ onOpenAnalysis }) => {
  const { reviewQueue } = useCoach();
  const [filter, setFilter] = useState("all");
  const filtered = useMemo(() => {
    if (filter === "all") return reviewQueue;
    return reviewQueue.filter((r) => r.urgency === filter);
  }, [filter]);

  return (
    <Section
      eyebrow="Coach queue"
      title="Videos awaiting your review"
      action={
        <div className="flex items-center gap-2 text-xs">
          {["all", "high", "medium", "low"].map((f) => (
            <button
              key={f}
              data-testid={`queue-filter-${f}`}
              onClick={() => setFilter(f)}
              className={`rounded-full border px-3 py-1.5 uppercase tracking-[0.14em] transition-all ${
                filter === f
                  ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200"
                  : "border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/20 hover:text-slate-200"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      }
    >
      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {filtered.map((item, i) => (
          <article
            key={item.id}
            data-testid={`queue-item-${item.id}`}
            className={`sv-rise sv-rise-${(i % 5) + 1} group overflow-hidden rounded-3xl border border-white/5 sv-glass transition-all duration-300 hover:-translate-y-1 hover:border-white/15 hover:shadow-[0_30px_60px_-25px_rgba(16,185,129,0.35)]`}
          >
            <div className="relative aspect-video overflow-hidden">
              <img
                src={item.thumbnail}
                alt={item.videoTitle}
                className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#05080F] via-[#05080F]/40 to-transparent" />
              <div className="absolute top-3 left-3 flex items-center gap-2">
                <span
                  className={`sv-chip ${
                    item.urgency === "high"
                      ? "sv-chip-gold"
                      : item.urgency === "medium"
                      ? "sv-chip-emerald"
                      : ""
                  }`}
                >
                  {item.urgency} priority
                </span>
                <span className="sv-chip font-mono">{item.duration}</span>
              </div>
              <button
                onClick={() => onOpenAnalysis(item)}
                data-testid={`queue-play-${item.id}`}
                className="absolute bottom-3 right-3 inline-flex h-11 w-11 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur-md transition-all hover:scale-105 hover:bg-emerald-400 hover:text-emerald-950"
              >
                <Play className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-col gap-4 p-5">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate font-display text-lg font-semibold text-slate-50">
                    {item.videoTitle}
                  </div>
                  <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                    <Clock className="h-3 w-3" />
                    Submitted {item.submittedAt}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                    AI Score
                  </div>
                  <div className="font-display text-2xl font-bold text-emerald-300">{item.aiScore}</div>
                </div>
              </div>
              <div className="flex items-center gap-3 border-t border-white/5 pt-4">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={item.avatar} alt={item.learner} />
                  <AvatarFallback className="bg-emerald-500/20 text-xs text-emerald-200">
                    {item.learner
                      .split(" ")
                      .map((s) => s[0])
                      .join("")}
                  </AvatarFallback>
                </Avatar>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-slate-100">{item.learner}</div>
                  <div className="text-xs text-slate-500">{item.tier}</div>
                </div>
                <button
                  data-testid={`queue-open-${item.id}`}
                  onClick={() => onOpenAnalysis(item)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-xs font-medium text-emerald-200 transition-all hover:border-emerald-400/60 hover:bg-emerald-400/20"
                >
                  Review <ArrowRight className="h-3 w-3" />
                </button>
              </div>
              <div className="grid grid-cols-3 gap-3 border-t border-white/5 pt-4">
                <ScoreBar label="Balance" value={item.scores.balance} />
                <ScoreBar label="Power" value={item.scores.power} tone="gold" />
                <ScoreBar label="Technique" value={item.scores.technique} />
              </div>
            </div>
          </article>
        ))}
      </div>
    </Section>
  );
};

/* -------------------------------- RANKINGS -------------------------------- */
const RankingsSection = () => {
  const { playerRankings } = useCoach();
  return (
    <Section
      eyebrow="Gamification"
      title="Player rankings"
      action={
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.02] px-3 py-1.5 text-xs text-slate-300">
          <Layers className="h-3.5 w-3.5 text-emerald-300" />
          Season 2026 · All squads
        </div>
      }
    >
      {/* Podium top-3 */}
      <div className="grid gap-5 md:grid-cols-3">
        {playerRankings.slice(0, 3).map((p, i) => {
          const podiumColor =
            i === 0
              ? "from-[#f5d982] to-[#D4AF37]"
              : i === 1
              ? "from-slate-200 to-slate-400"
              : "from-orange-300 to-amber-500";
          return (
            <div
              key={p.id}
              data-testid={`podium-${p.rank}`}
              className={`sv-rise sv-rise-${i + 1} relative overflow-hidden rounded-3xl border border-white/5 sv-glass p-6`}
            >
              <div
                className={`pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-gradient-to-br ${podiumColor} opacity-20 blur-3xl`}
              />
              <div className="relative flex items-center justify-between">
                <div
                  className={`inline-flex h-9 items-center gap-1.5 rounded-full bg-gradient-to-r ${podiumColor} px-3 text-xs font-semibold text-emerald-950`}
                >
                  {i === 0 ? <Crown className="h-3.5 w-3.5" /> : <Trophy className="h-3.5 w-3.5" />}
                  Rank #{p.rank}
                </div>
                <span className="flex items-center gap-1 text-xs text-slate-400">
                  {trendIcon(p.trend)}
                  {p.trendValue > 0 ? "+" : ""}
                  {p.trendValue}
                </span>
              </div>
              <div className="relative mt-6 flex items-center gap-4">
                <div className="relative">
                  <div
                    className={`h-16 w-16 overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br ${podiumColor} p-[2px]`}
                  >
                    <img src={p.avatar} alt={p.name} className="h-full w-full rounded-[14px] object-cover" />
                  </div>
                </div>
                <div>
                  <div className="font-display text-xl font-semibold text-slate-50">{p.name}</div>
                  <div className="text-xs text-slate-400">
                    {p.tier} · {p.specialty}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">{p.highlight}</div>
                </div>
              </div>
              <div className="relative mt-6 flex items-center justify-between border-t border-white/5 pt-4">
                <div>
                  <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">AI score</div>
                  <div className="font-display text-3xl font-bold text-slate-50">{p.aiScore}</div>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <Flame className="h-3.5 w-3.5 text-[#f5d982]" />
                    {p.streak}d streak
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Video className="h-3.5 w-3.5 text-emerald-300" />
                    {p.sessions} sessions
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Rest of leaderboard */}
      <div className="mt-6 overflow-hidden rounded-3xl border border-white/5 sv-glass">
        <div className="hidden grid-cols-[80px_1.6fr_1fr_1fr_1fr_1fr] items-center gap-4 border-b border-white/5 px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500 md:grid">
          <span>Rank</span>
          <span>Player</span>
          <span>AI Score</span>
          <span>Streak</span>
          <span>Sessions</span>
          <span className="text-right">Trend</span>
        </div>
        {playerRankings.slice(3).map((p, i) => (
          <div
            key={p.id}
            data-testid={`row-${p.rank}`}
            className={`grid grid-cols-2 items-center gap-4 border-b border-white/5 px-6 py-5 transition-colors last:border-b-0 hover:bg-white/[0.02] md:grid-cols-[80px_1.6fr_1fr_1fr_1fr_1fr] sv-rise sv-rise-${
              (i % 6) + 1
            }`}
          >
            <div className="col-span-1 flex items-center gap-2 text-sm text-slate-400">
              <span className="font-mono text-lg text-slate-200">#{p.rank}</span>
            </div>
            <div className="col-span-1 flex items-center gap-3 md:col-span-1">
              <Avatar className="h-9 w-9">
                <AvatarImage src={p.avatar} alt={p.name} />
                <AvatarFallback className="bg-emerald-500/20 text-xs text-emerald-200">
                  {p.name
                    .split(" ")
                    .map((s) => s[0])
                    .join("")}
                </AvatarFallback>
              </Avatar>
              <div>
                <div className="text-sm font-medium text-slate-100">{p.name}</div>
                <div className="text-xs text-slate-500">
                  {p.tier} · {p.specialty}
                </div>
              </div>
            </div>
            <div className="hidden md:block">
              <div className="font-display text-lg font-semibold text-emerald-300">{p.aiScore}</div>
            </div>
            <div className="hidden text-sm text-slate-300 md:block">{p.streak}d</div>
            <div className="hidden text-sm text-slate-300 md:block">{p.sessions}</div>
            <div className="hidden items-center justify-end gap-1.5 text-xs md:flex">
              {trendIcon(p.trend)}
              <span
                className={
                  p.trend === "up" ? "text-emerald-300" : p.trend === "down" ? "text-rose-300" : "text-slate-400"
                }
              >
                {p.trendValue > 0 ? "+" : ""}
                {p.trendValue}
              </span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
};

/* ------------------------------ TALENT SCOUTING ------------------------------ */
const TalentSection = () => {
  const { talentProspects, coach } = useCoach();
  const [selectedId, setSelectedId] = useState(talentProspects?.[0]?.id);
  const selected = talentProspects.find((p) => p.id === selectedId) || talentProspects[0];
  const [contactOpen, setContactOpen] = useState(false);
  const [message, setMessage] = useState(
    `Hello — we've been tracking your biomechanics scores this season and would love to invite you to the next-level trial. Coach ${coach.name}.`
  );

  if (!talentProspects || talentProspects.length === 0) {
    return (
      <Section
        eyebrow="Talent pipeline"
        title="Scouting the next generation"
      >
        <div className="rounded-3xl border border-white/5 sv-glass p-10 text-center">
          <Search className="mx-auto h-8 w-8 text-slate-600" />
          <div className="mt-4 font-display text-lg font-semibold text-slate-200">
            No prospects flagged yet
          </div>
          <p className="mx-auto mt-2 max-w-sm text-sm text-slate-400">
            Talent scouting surfaces once players build up enough session history for a trend to
            stand out. Check back after a few more weeks of reviews.
          </p>
        </div>
      </Section>
    );
  }

  const handleInvite = () => {
    setContactOpen(false);
    toast.success(`Invite sent to ${selected.name}`, {
      description: "They'll receive an email with the trial details within the hour.",
    });
  };

  return (
    <Section
      eyebrow="Talent pipeline"
      title="Scouting the next generation"
      action={
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.02] px-3 py-1.5 text-xs text-slate-300">
          <Search className="h-3.5 w-3.5 text-emerald-300" />
          {talentProspects.length} prospects flagged
        </div>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1fr,1.4fr]">
        <div className="flex flex-col gap-4">
          {talentProspects.map((p, i) => (
            <button
              key={p.id}
              data-testid={`prospect-${p.id}`}
              onClick={() => setSelectedId(p.id)}
              className={`sv-rise sv-rise-${i + 1} group flex items-center gap-4 rounded-2xl border p-4 text-left transition-all duration-300 ${
                selectedId === p.id
                  ? "border-emerald-400/40 bg-gradient-to-r from-emerald-500/15 via-emerald-500/5 to-transparent"
                  : "border-white/5 bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]"
              }`}
            >
              <div className="h-14 w-14 overflow-hidden rounded-2xl border border-white/10">
                <img src={p.avatar} alt={p.name} className="h-full w-full object-cover" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <div className="truncate font-display text-base font-semibold text-slate-50">
                    {p.name}
                  </div>
                  {p.nextLevelReady ? (
                    <BadgeCheck className="h-4 w-4 text-emerald-300" />
                  ) : null}
                </div>
                <div className="mt-0.5 text-xs text-slate-400">
                  {p.tier} · {p.style} · {p.hometown}
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Score</div>
                <div className="font-display text-xl font-bold text-emerald-300">{p.aiScore}</div>
              </div>
            </button>
          ))}
        </div>

        {/* Prospect detail */}
        <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass">
          <div className="relative h-40 w-full overflow-hidden">
            <img
              src="https://images.unsplash.com/photo-1780568521711-985c2118972a?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjY2NjV8MHwxfHNlYXJjaHwxfHxjcmlja2V0JTIwYmF0c21hbiUyMHNpbGhvdWV0dGV8ZW58MHx8fHwxNzgzMzE3ODU1fDA&ixlib=rb-4.1.0&q=85"
              alt="Batsman silhouette"
              className="h-full w-full object-cover opacity-70"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0C1322] via-[#0C1322]/70 to-transparent" />
          </div>
          <div className="-mt-14 px-6 pb-6">
            <div className="flex items-end justify-between gap-4">
              <div className="flex items-end gap-4">
                <div className="h-24 w-24 overflow-hidden rounded-3xl border border-white/10 bg-[#05080F] p-1">
                  <img src={selected.avatar} alt={selected.name} className="h-full w-full rounded-2xl object-cover" />
                </div>
                <div className="pb-2">
                  <div className="font-display text-2xl font-bold text-slate-50">{selected.name}</div>
                  <div className="text-xs text-slate-400">
                    {selected.tier} · Age {selected.age} · {selected.hometown}
                  </div>
                  {selected.contactedAt ? (
                    <div className="mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] text-emerald-200">
                      <Check className="h-3 w-3" />
                      Invited on {selected.contactedAt}
                    </div>
                  ) : null}
                </div>
              </div>
              <button
                data-testid="contact-prospect"
                onClick={() => setContactOpen(true)}
                className="sv-sheen inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-300 via-emerald-400 to-emerald-500 px-5 py-3 text-sm font-semibold text-emerald-950 shadow-[0_16px_40px_-16px_rgba(16,185,129,0.7)] transition-all hover:from-emerald-200 hover:via-emerald-300 hover:to-emerald-400"
              >
                <Mail className="h-4 w-4" />
                Contact / Invite to higher level
              </button>
            </div>

            <p className="mt-6 max-w-2xl text-sm leading-relaxed text-slate-400">
              <span className="font-medium text-slate-200">Signature.</span> {selected.signature}
            </p>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">AI Score</div>
                <div className="mt-2 font-display text-3xl font-bold text-emerald-300">
                  {selected.aiScore}
                </div>
                <div className="mt-1 text-xs text-slate-500">Peak {selected.peakScore}</div>
              </div>
              <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Style</div>
                <div className="mt-2 text-sm font-medium text-slate-100">{selected.style}</div>
              </div>
              <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Level ready</div>
                <div className="mt-2 flex items-center gap-1.5 text-sm font-medium text-slate-100">
                  {selected.nextLevelReady ? (
                    <>
                      <BadgeCheck className="h-4 w-4 text-emerald-300" />
                      State / National trial
                    </>
                  ) : (
                    <>
                      <Circle className="h-4 w-4 text-slate-500" />
                      Needs 4 more weeks
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-emerald-300">Strengths</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {selected.strengths.map((s) => (
                    <span key={s} className="sv-chip sv-chip-emerald">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-[#f5d982]">Focus areas</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {selected.weaknesses.map((s) => (
                    <span key={s} className="sv-chip sv-chip-gold">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {contactOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-md">
          <div
            data-testid="contact-modal"
            className="relative w-full max-w-xl overflow-hidden rounded-3xl border border-white/10 sv-glass-strong"
          >
            <div className="flex items-center justify-between border-b border-white/5 p-6">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                  Invite to higher level
                </div>
                <div className="mt-1 font-display text-lg font-semibold text-slate-50">
                  Contact {selected.name}
                </div>
              </div>
              <button
                data-testid="contact-close"
                onClick={() => setContactOpen(false)}
                className="rounded-full border border-white/10 bg-white/[0.02] p-2 text-slate-400 hover:text-slate-100"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-col gap-4 p-6">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Message
                </div>
                <Textarea
                  data-testid="contact-message"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  className="mt-1.5 min-h-[140px] rounded-xl border-white/10 bg-[#05080F] text-sm text-slate-200 focus-visible:ring-emerald-400/40"
                />
              </div>
              <div className="flex items-center justify-end gap-3">
                <button
                  data-testid="contact-cancel"
                  onClick={() => setContactOpen(false)}
                  className="rounded-xl border border-white/10 bg-white/[0.02] px-4 py-2.5 text-sm text-slate-300 hover:border-white/20 hover:bg-white/5"
                >
                  Cancel
                </button>
                <button
                  data-testid="contact-send"
                  onClick={handleInvite}
                  className="sv-sheen inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-300 via-emerald-400 to-emerald-500 px-5 py-2.5 text-sm font-semibold text-emerald-950 shadow-[0_16px_40px_-16px_rgba(16,185,129,0.7)]"
                >
                  <Send className="h-4 w-4" />
                  Send invite
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </Section>
  );
};

/* ------------------------------ VIDEO ANALYSIS ------------------------------ */
const AnalysisSection = ({ initial, onClear }) => {
  const { reviewQueue, biomechanicsSnapshot } = useCoach();
  const source = initial || reviewQueue[0];

  const [scores, setScores] = useState(source?.scores || { balance: 0, power: 0, technique: 0 });
  const [note, setNote] = useState("");
  const [playing, setPlaying] = useState(false);

  if (!source) {
    return (
      <Section eyebrow="AI-assisted review" title="Video analysis">
        <div className="rounded-3xl border border-white/5 sv-glass p-10 text-center text-sm text-slate-400">
          No session selected. Open a video from the review queue to analyze it.
        </div>
      </Section>
    );
  }

  const overall = Math.round((scores.balance + scores.power + scores.technique) / 3);

  const submit = () => {
    toast.success(`Feedback saved for ${source.learner}`, {
      description: "Player will see the updated scores and your notes in their feedback inbox.",
    });
  };

  return (
    <Section
      eyebrow="AI-assisted review"
      title={`Analysis · ${source.videoTitle}`}
      action={
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Avatar className="h-7 w-7">
            <AvatarImage src={source.avatar} alt={source.learner} />
            <AvatarFallback className="bg-emerald-500/20 text-[10px] text-emerald-200">
              {source.learner
                .split(" ")
                .map((s) => s[0])
                .join("")}
            </AvatarFallback>
          </Avatar>
          {source.learner} · {source.tier}
        </div>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1.5fr,1fr]">
        {/* Video with overlay */}
        <div className="relative overflow-hidden rounded-3xl border border-white/5 sv-glass">
          <div className="relative aspect-video">
            <img
              src={source.thumbnail}
              alt="video"
              className="h-full w-full object-cover"
            />
            {/* Wireframe overlay */}
            <svg viewBox="0 0 400 225" className="pointer-events-none absolute inset-0 h-full w-full opacity-70">
              <defs>
                <linearGradient id="wire" x1="0" x2="1" y1="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="100%" stopColor="#D4AF37" />
                </linearGradient>
              </defs>
              {/* Silhouette skeleton */}
              <g stroke="url(#wire)" strokeWidth="1.2" fill="none" opacity="0.9">
                <circle cx="200" cy="70" r="12" />
                <line x1="200" y1="82" x2="200" y2="130" />
                <line x1="200" y1="95" x2="176" y2="120" />
                <line x1="200" y1="95" x2="228" y2="120" />
                <line x1="176" y1="120" x2="164" y2="152" />
                <line x1="228" y1="120" x2="248" y2="150" />
                <line x1="200" y1="130" x2="184" y2="170" />
                <line x1="200" y1="130" x2="216" y2="170" />
                <line x1="184" y1="170" x2="180" y2="205" />
                <line x1="216" y1="170" x2="224" y2="205" />
              </g>
              {/* Ball trail */}
              <path
                d="M20 200 Q 120 40 220 130"
                stroke="#10b981"
                strokeWidth="1.5"
                fill="none"
                strokeDasharray="6 6"
                opacity="0.7"
              />
            </svg>

            {/* Floating biomechanics badges */}
            <div className="pointer-events-none absolute inset-0 p-4">
              <div className="flex flex-col gap-2">
                {[
                  { label: "Balance", value: scores.balance, color: "emerald" },
                  { label: "Power", value: scores.power, color: "gold" },
                  { label: "Technique", value: scores.technique, color: "emerald" },
                ].map((s) => (
                  <div
                    key={s.label}
                    className="w-fit rounded-xl border border-white/10 bg-[#0C1322]/70 px-3 py-2 backdrop-blur-md"
                  >
                    <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                      {s.label}
                    </div>
                    <div className="font-display text-lg font-bold text-slate-50">
                      {s.value}
                      <span className="ml-1 text-[10px] font-medium text-slate-500">/100</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Timeline / controls */}
            <div className="absolute inset-x-4 bottom-4 rounded-2xl border border-white/10 bg-[#05080F]/70 p-3 backdrop-blur-xl">
              <div className="flex items-center gap-3">
                <button
                  data-testid="video-rewind"
                  className="rounded-full border border-white/10 bg-white/5 p-2 text-slate-200 hover:bg-white/10"
                >
                  <Rewind className="h-4 w-4" />
                </button>
                <button
                  data-testid="video-play"
                  onClick={() => setPlaying((p) => !p)}
                  className="rounded-full bg-emerald-400 p-2 text-emerald-950 hover:bg-emerald-300"
                >
                  {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <div className="flex-1">
                  <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                    <div className="absolute inset-y-0 left-0 w-[42%] rounded-full bg-gradient-to-r from-emerald-300 to-[#D4AF37]" />
                    <div className="absolute left-[42%] top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow" />
                  </div>
                </div>
                <div className="font-mono text-xs text-slate-300">00:18 / {source.duration}</div>
              </div>
            </div>
          </div>

          <div className="grid gap-4 border-t border-white/5 p-6 md:grid-cols-3">
            {Object.entries(biomechanicsSnapshot).map(([k, v]) => (
              <div key={k}>
                <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                  {k} · breakdown
                </div>
                <div className="mt-3 flex flex-col gap-2">
                  {v.breakdown.map((b) => (
                    <ScoreBar
                      key={b.label}
                      label={b.label}
                      value={b.value}
                      tone={k === "power" ? "gold" : "emerald"}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Override + notes panel */}
        <div className="flex flex-col gap-5">
          <div className="rounded-3xl border border-white/5 sv-glass p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
                  AI overall
                </div>
                <div className="mt-1 font-display text-4xl font-bold text-slate-50">{overall}</div>
              </div>
              <div className="text-right">
                <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                  Coach confidence
                </div>
                <div className="mt-1 flex items-center justify-end gap-1 text-xs text-slate-300">
                  <Gauge className="h-3.5 w-3.5 text-emerald-300" />
                  High
                </div>
              </div>
            </div>
            <div className="mt-5 flex flex-col gap-4">
              {["balance", "power", "technique"].map((k) => (
                <div key={k}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                      Override {k}
                    </span>
                    <span className="font-mono text-slate-200">{scores[k]}</span>
                  </div>
                  <input
                    data-testid={`override-${k}`}
                    type="range"
                    min="0"
                    max="100"
                    value={scores[k]}
                    onChange={(e) => setScores((s) => ({ ...s, [k]: Number(e.target.value) }))}
                    className="mt-2 h-1.5 w-full appearance-none rounded-full bg-white/10 accent-emerald-400"
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-white/5 sv-glass p-6">
            <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
              Coach recommendation
            </div>
            <div className="mt-1 font-display text-lg font-semibold text-slate-50">
              Manual notes & drills
            </div>
            <Textarea
              data-testid="coach-note"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Add specific drills, cues, or corrections for the player…"
              className="mt-4 min-h-[140px] rounded-xl border-white/10 bg-[#05080F] text-sm text-slate-200 focus-visible:ring-emerald-400/40"
            />
            <div className="mt-4 flex items-center justify-between gap-3">
              <button
                data-testid="mark-reviewed"
                onClick={onClear}
                className="rounded-xl border border-white/10 bg-white/[0.02] px-4 py-2.5 text-sm text-slate-300 hover:border-white/20 hover:bg-white/5"
              >
                Skip for now
              </button>
              <button
                data-testid="submit-feedback"
                onClick={submit}
                className="sv-sheen inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-300 via-emerald-400 to-emerald-500 px-5 py-2.5 text-sm font-semibold text-emerald-950 shadow-[0_16px_40px_-16px_rgba(16,185,129,0.7)]"
              >
                <Check className="h-4 w-4" />
                Send feedback
              </button>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
};

/* --------------------------------- PAGE --------------------------------- */
const CoachDashboard = () => {
  const [active, setActive] = useState("overview");
  const [analysisTarget, setAnalysisTarget] = useState(null);

  const openAnalysis = (item) => {
    setAnalysisTarget(item);
    setActive("analysis");
  };

  const titleMap = {
    overview: "Command Center",
    queue: "Review Queue",
    rankings: "Player Rankings",
    talent: "Talent Scouting",
    analysis: "Video Analysis",
  };

  const { data, isLoading, error } = useQuery({
    queryKey: ["coachDashboard"],
    queryFn: () => fetchWithAuth("/coach/me/"),
  });

  if (isLoading || !data) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#05080F] text-emerald-400">
        Loading your academy...
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#05080F] text-red-400">
        Error loading dashboard: {error.message}
      </div>
    );
  }

  const academyName = data.academy?.name || "Your Academy";
  const players = data.players || [];
  const sessions = data.review_queue || [];

  const coach = {
    name: academyName,
    role: "coach",
    avatar: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?crop=faces&fit=crop&w=200&h=200",
    email: "",
  };

  const playerById = Object.fromEntries(players.map((p) => [p.id, p]));

  const reviewQueue = sessions.map((s) => {
    const player = playerById[s.player] || {};
    return {
      id: s.id,
      learner: player.name || "Player",
      tier: player.playing_level || "—",
      avatar: "https://images.unsplash.com/photo-1607746882042-944635dfe10e?crop=faces&fit=crop&w=256&h=256",
      thumbnail: "https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=800",
      duration: "0:30",
      urgency: s.status === "COMPLETED" ? "low" : "high",
      submittedAt: new Date(s.date_analyzed).toLocaleDateString(),
      videoTitle: s.title || `Session ${s.id}`,
      aiScore: s.overall_score,
      scores: { balance: s.balance_score, power: s.power_score, technique: s.technique_score },
    };
  });

  const avgScore = sessions.length
    ? Math.round(sessions.reduce((sum, s) => sum + (s.overall_score || 0), 0) / sessions.length)
    : 0;

  const coachMetrics = [
    { label: "Total players", value: players.length, delta: "enrolled", tone: "emerald", testId: "metric-players" },
    { label: "Sessions reviewed", value: sessions.length, delta: "all time", tone: "emerald", testId: "metric-sessions" },
    { label: "Avg AI score", value: avgScore || "—", delta: "across all sessions", tone: "gold", testId: "metric-avg" },
    {
      label: "Awaiting review",
      value: sessions.filter((s) => s.status !== "COMPLETED").length,
      delta: "in queue",
      tone: "muted",
      testId: "metric-queue",
    },
  ];

  const progressTimeline = [...sessions]
    .slice()
    .reverse()
    .slice(-12)
    .map((s, i) => ({
      week: `S${i + 1}`,
      balance: s.balance_score,
      power: s.power_score,
      technique: s.technique_score,
    }));

  const recentActivity = sessions.slice(0, 6).map((s) => {
    const player = playerById[s.player] || {};
    return {
      id: s.id,
      icon: "video",
      text: `${player.name || "A player"} submitted "${s.title || "a session"}"`,
      time: new Date(s.date_analyzed).toLocaleDateString(),
    };
  });

  const latest = sessions[0];
  const latestPlayer = latest ? playerById[latest.player] || {} : null;
  const latestSession = latest
    ? {
        learner: latestPlayer.name || "A player",
        title: latest.title || `Session ${latest.id}`,
        date: new Date(latest.date_analyzed).toLocaleDateString(),
      }
    : null;

  const emptyMetric = { score: 0, trend: "—", breakdown: [] };
  const biomechanicsSnapshot = latest
    ? {
        balance: { score: latest.balance_score, trend: "—", breakdown: [{ label: "Overall", value: latest.balance_score }] },
        power: { score: latest.power_score, trend: "—", breakdown: [{ label: "Overall", value: latest.power_score }] },
        technique: { score: latest.technique_score, trend: "—", breakdown: [{ label: "Overall", value: latest.technique_score }] },
      }
    : { balance: emptyMetric, power: emptyMetric, technique: emptyMetric };

  const playerRankings = [...players]
    .sort((a, b) => (b.total_points || 0) - (a.total_points || 0))
    .map((p, i) => {
      const playerSessions = sessions.filter((s) => s.player === p.id);
      const best = playerSessions.reduce((m, s) => Math.max(m, s.overall_score || 0), 0);
      return {
        id: p.id,
        rank: i + 1,
        name: p.name,
        avatar: "https://images.unsplash.com/photo-1607746882042-944635dfe10e?crop=faces&fit=crop&w=256&h=256",
        tier: p.playing_level || "—",
        specialty: p.batting_hand ? `${p.batting_hand}-handed` : "—",
        highlight: p.archetype || "",
        aiScore: best,
        peakScore: best,
        trend: "flat",
        trendValue: 0,
        streak: p.current_streak || 0,
        sessions: playerSessions.length,
      };
    });

  const talentProspects = playerRankings
    .filter((p) => p.aiScore >= 85)
    .map((p) => ({
      ...p,
      age: "—",
      hometown: academyName,
      style: p.specialty,
      signature: `Consistently scoring ${p.aiScore}+ across ${p.sessions} session${p.sessions === 1 ? "" : "s"}.`,
      strengths: [],
      weaknesses: [],
      nextLevelReady: p.aiScore >= 90,
    }));

  const coachSessions = []; // no coaching-calendar backend yet

  const contextValue = {
    coach,
    reviewQueue,
    coachMetrics,
    progressTimeline,
    coachSessions,
    recentActivity,
    biomechanicsSnapshot,
    playerRankings,
    talentProspects,
    latestSession,
  };

  return (
    <CoachContext.Provider value={contextValue}>
      <AppShell role="coach" activeKey={active} onNavigate={setActive} title={titleMap[active]} profile={coach}>
        {active === "overview" ? <OverviewSection onOpenAnalysis={openAnalysis} /> : null}
        {active === "queue" ? <ReviewQueueSection onOpenAnalysis={openAnalysis} /> : null}
        {active === "rankings" ? <RankingsSection /> : null}
        {active === "talent" ? <TalentSection /> : null}
        {active === "analysis" ? (
          <AnalysisSection initial={analysisTarget} onClear={() => setActive("queue")} />
        ) : null}
      </AppShell>
    </CoachContext.Provider>
  );
};

export default CoachDashboard;

export const Route = createFileRoute("/coach")({
  component: CoachDashboard,
});
