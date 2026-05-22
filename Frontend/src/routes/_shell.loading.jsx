import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

export const Route = createFileRoute("/_shell/loading")({
  component: LoadingPage,
});

const PHASES = [
  "Extracting frames…",
  "Running MediaPipe pose estimation…",
  "Analyzing biomechanics…",
  "Scoring 4-factor breakdown…",
  "Finalizing report…",
];

function LoadingPage() {
  const [i, setI] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const t = setInterval(() => setI((n) => (n + 1) % PHASES.length), 1100);
    const done = setTimeout(() => navigate({ to: "/results" }), 5500);
    return () => { clearInterval(t); clearTimeout(done); };
  }, [navigate]);

  return (
    <div className="px-6 lg:px-10 py-8 max-w-[1400px] mx-auto">
      <header className="mb-8 flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-primary/15 text-primary grid place-items-center animate-pulse">
          <Activity className="h-5 w-5" />
        </div>
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Processing</p>
          <h1 className="text-2xl font-semibold tracking-tight">{PHASES[i]}</h1>
        </div>
      </header>

      {/* Skeleton dashboard mirroring results layout */}
      <div className="grid lg:grid-cols-[1.1fr_1.2fr_1fr] gap-6">
        <div className="rounded-2xl border border-border bg-card p-5 space-y-4">
          <Skeleton className="aspect-video w-full rounded-xl" />
          <div className="grid grid-cols-7 gap-2">
            {Array.from({ length: 7 }).map((_, k) => (
              <Skeleton key={k} className="aspect-square rounded-md" />
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, k) => (
            <div key={k} className="rounded-2xl border border-border bg-card p-5 space-y-3">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-7 w-3/4" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-5/6" />
            </div>
          ))}
        </div>

        <div className="rounded-2xl border border-border bg-card p-5 space-y-5">
          {Array.from({ length: 4 }).map((_, k) => (
            <div key={k} className="space-y-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-2.5 w-full rounded-full" />
            </div>
          ))}
          <Skeleton className="aspect-square w-full rounded-xl" />
        </div>
      </div>

      <div className="mt-8 flex items-center gap-2 text-xs text-muted-foreground">
        {PHASES.map((p, idx) => (
          <span key={p} className={
            "px-2.5 py-1 rounded-full border " +
            (idx === i ? "border-primary text-primary bg-primary/10" :
             idx < i ? "border-border text-muted-foreground" :
             "border-border/60 text-muted-foreground/60")
          }>{p}</span>
        ))}
      </div>
    </div>
  );
}
