import { createFileRoute } from "@tanstack/react-router";
import { useState, useMemo } from "react";
import { Play, AlertTriangle, ArrowRight, TrendingUp } from "lucide-react";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

export const Route = createFileRoute("/_shell/compare")({
  component: ComparePage,
});

const SESSIONS = [
  { id: "s1", date: "Apr 14, 2025", score: 71, balance: 64, weakness: "Front foot across off-stump" },
  { id: "s2", date: "May 09, 2025", score: 76, balance: 70, weakness: "Closed front shoulder on drive" },
  { id: "s3", date: "May 21, 2025", score: 79, balance: 73, weakness: "Late trigger movement" },
  { id: "s4", date: "Jun 03, 2025", score: 81, balance: 75, weakness: "Bat tap timing inconsistent" },
  { id: "s5", date: "Jun 18, 2025", score: 82, balance: 78, weakness: "Front pad falls across off-stump" },
];

function ComparePage() {
  const [a, setA] = useState("s2");
  const [b, setB] = useState("s5");

  const sa = SESSIONS.find((s) => s.id === a);
  const sb = SESSIONS.find((s) => s.id === b);

  const delta = useMemo(() => sb.balance - sa.balance, [sa, sb]);

  return (
    <div className="px-6 lg:px-10 py-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground mb-2">Session compare</p>
        <h1 className="text-3xl lg:text-4xl font-semibold tracking-tight">
          Player A — A vs B
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Compare two sessions side-by-side to track tangible progress.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-center mb-6">
        <SessionPicker label="Session A" value={a} onChange={setA} />
        <ArrowRight className="hidden md:block h-5 w-5 text-muted-foreground mx-auto" />
        <SessionPicker label="Session B" value={b} onChange={setB} align="right" />
      </div>

      <div className="grid md:grid-cols-[1fr_auto_1fr] gap-6 items-stretch">
        <SessionPanel session={sa} side="left" />

        <div className="md:flex flex-col items-center justify-center hidden">
          <div className="rounded-2xl border border-strength/40 bg-strength/10 px-5 py-6 text-center max-w-[200px]">
            <TrendingUp className="h-6 w-6 mx-auto text-strength" />
            <div className="text-[11px] uppercase tracking-wider text-strength/80 mt-2">Improvement</div>
            <div className="text-3xl font-bold tabular-nums text-strength mt-1">
              {delta >= 0 ? "+" : ""}{delta}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">Balance</div>
          </div>

          <div className="mt-4 text-center">
            <div className="text-xs text-muted-foreground">Overall</div>
            <div className="text-lg font-semibold tabular-nums">
              {sa.score} <ArrowRight className="inline h-3.5 w-3.5 mx-1 text-muted-foreground" /> {sb.score}
            </div>
          </div>
        </div>

        <SessionPanel session={sb} side="right" />
      </div>
    </div>
  );
}

function SessionPicker({ label, value, onChange, align }) {
  return (
    <div className={align === "right" ? "md:text-right" : ""}>
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1.5">{label}</div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-11 bg-input/40">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SESSIONS.map((s) => (
            <SelectItem key={s.id} value={s.id}>
              {s.date} · score {s.score}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function SessionPanel({ session, side }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 space-y-5">
      <div className="aspect-video rounded-xl border border-border bg-gradient-to-br from-muted to-background grid place-items-center relative overflow-hidden">
        <button className="h-12 w-12 rounded-full bg-primary text-primary-foreground grid place-items-center shadow-xl shadow-primary/30">
          <Play className="h-5 w-5 ml-0.5" />
        </button>
        <div className="absolute top-3 left-3 text-[11px] font-medium px-2 py-1 rounded-md bg-background/80 backdrop-blur border border-border">
          {side === "left" ? "Session A" : "Session B"} · {session.date}
        </div>
      </div>

      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Overall score</div>
          <div className="text-5xl font-bold tabular-nums mt-1">{session.score}</div>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Balance</div>
          <div className="text-2xl font-semibold tabular-nums">{session.balance}</div>
        </div>
      </div>

      <div className="rounded-xl border border-weakness/40 bg-weakness/5 p-4 relative overflow-hidden">
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-weakness" />
        <div className="flex items-center gap-2 mb-2">
          <div className="h-7 w-7 rounded-lg bg-weakness/15 text-weakness grid place-items-center">
            <AlertTriangle className="h-3.5 w-3.5" />
          </div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-weakness">
            Primary weakness
          </span>
        </div>
        <h3 className="text-base font-semibold tracking-tight leading-snug">
          {session.weakness}
        </h3>
      </div>
    </div>
  );
}
