import { createFileRoute, Link, useLocation } from "@tanstack/react-router";
import { useState } from "react";
import { Play, AlertTriangle, ShieldCheck, Wrench, Sparkles, GitCompareArrows } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/_shell/results")({
  component: ResultsPage,
});

function ResultsPage() {
  const [frame, setFrame] = useState(0);
  const location = useLocation();
  const apiData = location.state?.apiData || null;

  if (!apiData) {
    return (
      <div className="px-6 py-20 text-center">
        <h2 className="text-2xl font-semibold mb-4">No Analysis Data Found</h2>
        <p className="text-muted-foreground mb-6">Please run a new session from the Analyzer.</p>
        <Link to="/analyzer">
          <Button>Go to Analyzer</Button>
        </Link>
      </div>
    );
  }

  const { scores, frame_urls, session_id } = apiData;

  const dynamicFactors = [
    {
      key: "weakness",
      label: "Weakness & Vulnerability",
      icon: AlertTriangle,
      accent: "weakness",
      title: "Primary Weakness Identified",
      body: scores.primary_weakness || "None identified.",
    },
    {
      key: "strength",
      label: "Strength & Power",
      icon: ShieldCheck,
      accent: "strength",
      title: "Primary Strength Identified",
      body: scores.primary_strength || "None identified.",
    },
    {
      key: "change",
      label: "Thing to Change · Drills",
      icon: Wrench,
      accent: "change",
      title: "AI Coaching Recommendation",
      body: "Focus on balancing your weight transfer based on the scores provided above. Work closely with a coach to review this AI feedback.",
    },
    {
      key: "bonus",
      label: "Bonus Insight",
      icon: Sparkles,
      accent: "bonus",
      title: "Pro Comparison",
      body: "Your power score indicates strong core rotation, similar to modern T20 power hitters.",
    },
  ];

  const dynamicMetrics = [
    { label: "Balance",   value: scores.balance_score || 0 },
    { label: "Power",     value: scores.power_score || 0 },
    { label: "Technique", value: scores.technique_score || 0 },
    { label: "Defence",   value: scores.defence_score || 0 },
  ];

  return (
    <div className="px-6 lg:px-10 py-8 max-w-[1500px] mx-auto">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-full bg-muted grid place-items-center text-sm font-semibold">P1</div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Session #{session_id}</p>
            <h1 className="text-2xl lg:text-3xl font-semibold tracking-tight">AI Stance Report</h1>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="rounded-full bg-bonus/15 text-bonus text-xs font-medium px-3 py-1.5">
            Analyzed by Gemini
          </span>
          <div className="text-right">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Overall</div>
            <div className="flex items-baseline gap-1">
              <span className="text-5xl font-bold tabular-nums leading-none">{scores.overall_score || 0}</span>
              <span className="text-sm text-muted-foreground">/100</span>
            </div>
          </div>
          <Link to="/compare">
            <Button variant="outline" className="gap-2">
              <GitCompareArrows className="h-4 w-4" /> Compare
            </Button>
          </Link>
        </div>
      </header>

      <div className="grid lg:grid-cols-[1.1fr_1.3fr_1fr] gap-6">
        {/* Left: Video + frames */}
        <section className="rounded-2xl border border-border bg-card p-5">
          <div className="aspect-video rounded-xl bg-gradient-to-br from-muted to-background border border-border grid place-items-center relative overflow-hidden">
            {frame_urls && frame_urls[frame] ? (
              <img 
                src={`http://localhost:8000${frame_urls[frame]}`} 
                alt="Extracted frame" 
                className="absolute inset-0 w-full h-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_40%,_oklch(0.78_0.16_155/_0.15),transparent_60%)]" />
            )}
            
            <div className="absolute bottom-3 left-3 right-3 flex items-center gap-2 text-xs text-foreground bg-background/60 backdrop-blur-md px-3 py-1.5 rounded-full">
              <span className="font-medium">Frame {frame + 1}/4</span>
              <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary" style={{ width: `${((frame + 1) / 4) * 100}%` }} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2 mt-4">
            {frame_urls && frame_urls.map((url, k) => {
              const sel = k === frame;
              return (
                <button
                  key={k}
                  onClick={() => setFrame(k)}
                  className={
                    "aspect-square rounded-md border bg-muted relative overflow-hidden " +
                    (sel ? "border-primary ring-2 ring-primary/30" : "border-border hover:border-border/80")
                  }
                >
                  <img src={`http://localhost:8000${url}`} className="w-full h-full object-cover opacity-80" alt={`Thumbnail ${k+1}`} />
                  <span className="absolute bottom-1 left-1 text-[10px] font-mono text-white bg-black/50 px-1 rounded">
                    {["Pre-stance", "Backlift", "Impact", "Follow-through"][k]}
                  </span>
                </button>
              );
            })}
          </div>

          <p className="text-xs text-muted-foreground mt-4">
            Key biomechanical frames extracted by Gemini Flash Timeline Locator.
          </p>
        </section>

        {/* Middle: 4 Factors */}
        <section className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {dynamicFactors.map((f) => <FactorCard key={f.key} {...f} />)}
        </section>

        {/* Right: Metrics + heatmap */}
        <section className="rounded-2xl border border-border bg-card p-5">
          <div className="text-xs uppercase tracking-wider text-muted-foreground mb-4">Metrics</div>
          <div className="space-y-4">
            {dynamicMetrics.map((m) => (
              <div key={m.label}>
                <div className="flex items-baseline justify-between mb-1.5">
                  <span className="text-sm font-medium">{m.label}</span>
                  <span className="text-sm font-semibold tabular-nums">{m.value}</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary" style={{ width: `${m.value}%` }} />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-3">
              Delivery vulnerability heatmap
            </div>
            <Heatmap />
            <div className="flex items-center justify-between text-[11px] text-muted-foreground mt-2">
              <span>Safe</span>
              <span>Vulnerable</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function FactorCard({ icon: Icon, accent, label, title, body }) {
  // map accent token to classes — using CSS variables via inline style for the bar
  const accentVar = `var(--color-${accent})`;
  return (
    <div className="rounded-2xl border border-border bg-card p-5 relative overflow-hidden">
      <div
        className="absolute left-0 top-0 bottom-0 w-1"
        style={{ background: accentVar }}
      />
      <div className="flex items-center gap-2 mb-3">
        <div
          className="h-8 w-8 rounded-lg grid place-items-center"
          style={{ background: `color-mix(in oklab, ${accentVar} 18%, transparent)`, color: accentVar }}
        >
          <Icon className="h-4 w-4" />
        </div>
        <span
          className="text-[11px] font-semibold uppercase tracking-wider"
          style={{ color: accentVar }}
        >
          {label}
        </span>
      </div>
      <h3 className="text-base font-semibold tracking-tight leading-snug">{title}</h3>
      <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{body}</p>
    </div>
  );
}

function Heatmap() {
  // 6x6 grid representing delivery zones, vulnerability values 0..1
  const cells = Array.from({ length: 36 }, (_, k) => {
    const x = k % 6;
    const y = Math.floor(k / 6);
    // hot zone around top-right (off-stump short)
    const dx = x - 4.5, dy = y - 1;
    const v = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / 4);
    return v;
  });
  return (
    <div className="aspect-square rounded-xl border border-border overflow-hidden grid grid-cols-6">
      {cells.map((v, k) => (
        <div
          key={k}
          className="border border-border/40"
          style={{ background: `color-mix(in oklab, var(--color-weakness) ${Math.round(v * 85)}%, var(--color-muted))` }}
          title={`Zone ${k + 1} · vulnerability ${(v * 100).toFixed(0)}%`}
        />
      ))}
    </div>
  );
}
