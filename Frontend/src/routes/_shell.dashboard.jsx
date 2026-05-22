import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { ChevronDown, Plus, Calendar, Play, TrendingUp, Trophy, Target, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

export const Route = createFileRoute("/_shell/dashboard")({
  component: DashboardPage,
});

const PLAYERS = [
  { id: "a", name: "Player A", initials: "PA", role: "Top order", score: 82, trend: +4 },
  { id: "b", name: "Player B", initials: "PB", role: "Middle order", score: 74, trend: -2 },
  { id: "c", name: "Player C", initials: "PC", role: "Opener", score: 88, trend: +6 },
  { id: "d", name: "Player D", initials: "PD", role: "All-rounder", score: 69, trend: +1 },
  { id: "e", name: "Player E", initials: "PE", role: "Finisher", score: 77, trend: +3 },
];

const TREND = [
  { date: "Apr 02", score: 68 },
  { date: "Apr 14", score: 71 },
  { date: "Apr 28", score: 70 },
  { date: "May 09", score: 76 },
  { date: "May 21", score: 79 },
  { date: "Jun 03", score: 81 },
  { date: "Jun 18", score: 82 },
];

const SESSIONS = [
  { id: 1, date: "Jun 18, 2025", duration: "00:42", deliveries: 12, score: 82, archetype: "Anchor" },
  { id: 2, date: "Jun 03, 2025", duration: "00:51", deliveries: 16, score: 81, archetype: "Anchor" },
  { id: 3, date: "May 21, 2025", duration: "00:38", deliveries: 10, score: 79, archetype: "Counter-puncher" },
  { id: 4, date: "May 09, 2025", duration: "00:46", deliveries: 14, score: 76, archetype: "Anchor" },
];

function DashboardPage() {
  const [active, setActive] = useState(PLAYERS[0]);

  return (
    <div className="px-6 lg:px-10 py-8 max-w-[1400px] mx-auto">
      <header className="flex flex-wrap items-end justify-between gap-4 mb-8">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground mb-2">Coach console</p>
          <h1 className="text-3xl lg:text-4xl font-semibold tracking-tight">Academy Overview</h1>
          <p className="text-sm text-muted-foreground mt-1">
            5 active players · 24 sessions analyzed this month
          </p>
        </div>

        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="h-10 gap-2">
                <span className="h-6 w-6 rounded-full bg-primary/20 text-primary text-[10px] grid place-items-center font-semibold">
                  {active.initials}
                </span>
                Switch player: <span className="font-semibold">{active.name}</span>
                <ChevronDown className="h-4 w-4 opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              {PLAYERS.map((p) => (
                <DropdownMenuItem key={p.id} onClick={() => setActive(p)}>
                  <span className="h-6 w-6 rounded-full bg-muted text-[10px] grid place-items-center font-semibold mr-2">
                    {p.initials}
                  </span>
                  {p.name}
                  <span className="ml-auto text-xs text-muted-foreground">{p.score}</span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          <Link to="/analyzer">
            <Button className="h-10 gap-2">
              <Plus className="h-4 w-4" /> New analysis
            </Button>
          </Link>
        </div>
      </header>

      {/* Roster */}
      <section className="mb-10">
        <SectionTitle icon={Trophy} title="Roster" hint="Tap a profile to switch the dashboard" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {PLAYERS.map((p) => {
            const selected = p.id === active.id;
            return (
              <button
                key={p.id}
                onClick={() => setActive(p)}
                className={
                  "text-left rounded-2xl border p-4 transition-all " +
                  (selected
                    ? "border-primary/50 bg-primary/[0.06] shadow-[0_0_0_1px_var(--color-primary)]"
                    : "border-border bg-card hover:border-border/80 hover:bg-card/80")
                }
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="h-10 w-10 rounded-full bg-muted grid place-items-center text-sm font-semibold">
                    {p.initials}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-semibold truncate">{p.name}</div>
                    <div className="text-[11px] text-muted-foreground truncate">{p.role}</div>
                  </div>
                </div>
                <div className="flex items-baseline justify-between">
                  <div className="text-2xl font-bold tabular-nums">{p.score}</div>
                  <div className={
                    "text-xs font-medium tabular-nums " +
                    (p.trend >= 0 ? "text-strength" : "text-weakness")
                  }>
                    {p.trend >= 0 ? "+" : ""}{p.trend}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Trend */}
      <section className="mb-10">
        <SectionTitle icon={TrendingUp} title={`${active.name} · Stance score trend`} hint="Last 7 sessions" />
        <div className="rounded-2xl border border-border bg-card p-5">
          <div className="flex items-baseline gap-3 mb-4">
            <div className="text-4xl font-bold tabular-nums">{active.score}</div>
            <div className="text-xs text-muted-foreground">/ 100 latest overall</div>
            <div className={
              "ml-auto text-xs font-medium tabular-nums " +
              (active.trend >= 0 ? "text-strength" : "text-weakness")
            }>
              {active.trend >= 0 ? "▲" : "▼"} {Math.abs(active.trend)} pts vs last
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={TREND} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
                <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} domain={[60, 90]} />
                <Tooltip
                  contentStyle={{
                    background: "var(--color-popover)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Line type="monotone" dataKey="score" stroke="var(--color-primary)" strokeWidth={2.5} dot={{ r: 3, fill: "var(--color-primary)" }} activeDot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Sessions table */}
      <section>
        <SectionTitle icon={Calendar} title="Past sessions" hint={`Uploads for ${active.name}`} />
        <div className="rounded-2xl border border-border bg-card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-xs uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="text-left font-medium px-5 py-3">Date</th>
                <th className="text-left font-medium px-5 py-3">Duration</th>
                <th className="text-left font-medium px-5 py-3">Deliveries</th>
                <th className="text-left font-medium px-5 py-3">Archetype</th>
                <th className="text-left font-medium px-5 py-3">Score</th>
                <th className="text-right font-medium px-5 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {SESSIONS.map((s) => (
                <tr key={s.id} className="border-t border-border hover:bg-muted/20 transition-colors">
                  <td className="px-5 py-4 font-medium">{s.date}</td>
                  <td className="px-5 py-4 text-muted-foreground tabular-nums">{s.duration}</td>
                  <td className="px-5 py-4 text-muted-foreground tabular-nums">{s.deliveries}</td>
                  <td className="px-5 py-4">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-bonus/15 text-bonus px-2.5 py-1 text-[11px] font-medium">
                      <Target className="h-3 w-3" /> {s.archetype}
                    </span>
                  </td>
                  <td className="px-5 py-4 font-semibold tabular-nums">{s.score}</td>
                  <td className="px-5 py-4 text-right">
                    <Link to="/results">
                      <Button size="sm" variant="ghost" className="h-8 gap-1.5">
                        <Play className="h-3.5 w-3.5" /> View
                      </Button>
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function SectionTitle({ icon: Icon, title, hint }) {
  return (
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
      </div>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  );
}
