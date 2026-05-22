import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { UploadCloud, Check, ChevronRight, Sparkles, Film } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

export const Route = createFileRoute("/_shell/analyzer")({
  component: AnalyzerPage,
});

const STEPS = [
  { id: 1, key: "profile", label: "Profile" },
  { id: 2, key: "stance",  label: "Stance" },
  { id: 3, key: "grip",    label: "Grip" },
  { id: 4, key: "review",  label: "Review" },
];

function AnalyzerPage() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [data, setData] = useState({
    hand: "right", level: "club",
    foot: "side-on", width: "shoulder", guard: "middle",
    grip: "v", backlift: "second-slip",
  });
  const navigate = useNavigate();

  const set = (k, v) => setData((d) => ({ ...d, [k]: v }));

  return (
    <div className="px-6 lg:px-10 py-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground mb-2">New analysis</p>
        <h1 className="text-3xl lg:text-4xl font-semibold tracking-tight">Upload & configure session</h1>
        <p className="text-sm text-muted-foreground mt-1">
          A side-on, well-lit clip of the batter taking guard works best.
        </p>
      </header>

      <div className="grid lg:grid-cols-[1.05fr_1fr] gap-6">
        {/* Left: Upload */}
        <div className="rounded-2xl border border-border bg-card p-5">
          <div className="text-xs uppercase tracking-wider text-muted-foreground mb-3">Video</div>

          <label
            htmlFor="video"
            className="block rounded-xl border border-dashed border-border bg-muted/20 hover:bg-muted/30 transition-colors cursor-pointer"
          >
            <div className="aspect-video grid place-items-center text-center px-6">
              {file ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="h-14 w-14 rounded-full bg-primary/15 grid place-items-center">
                    <Film className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">{file.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(1)} MB · ready to analyze
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="h-14 w-14 rounded-full bg-muted grid place-items-center">
                    <UploadCloud className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">Drop your video here</div>
                    <div className="text-xs text-muted-foreground">MP4 or MOV · up to 200 MB</div>
                  </div>
                  <Button type="button" variant="outline" size="sm" className="mt-1">Browse files</Button>
                </div>
              )}
            </div>
            <input
              id="video"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>

          <div className="grid grid-cols-3 gap-3 mt-5">
            {[
              { k: "Angle", v: "Side-on" },
              { k: "Lighting", v: "Even" },
              { k: "Frame", v: "Full body" },
            ].map((t) => (
              <div key={t.k} className="rounded-xl border border-border p-3 min-w-0">
                <div className="text-[11px] uppercase tracking-wider text-muted-foreground truncate">{t.k}</div>
                <div className="text-sm font-medium mt-0.5 truncate">{t.v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Stepper */}
        <div className="rounded-2xl border border-border bg-card p-5 flex flex-col">
          <Stepper step={step} />

          <div className="mt-6 flex-1">
            {step === 1 && (
              <Group title="Player profile">
                <SelectField label="Batting hand" value={data.hand} onChange={(v) => set("hand", v)}
                  options={[["right","Right-handed"],["left","Left-handed"]]} />
                <SelectField label="Playing level" value={data.level} onChange={(v) => set("level", v)}
                  options={[["junior","Junior / U-16"],["club","Club / District"],["state","State / Ranji"],["pro","Professional"]]} />
              </Group>
            )}
            {step === 2 && (
              <Group title="Stance setup">
                <ButtonGrid label="Foot position" value={data.foot} onChange={(v) => set("foot", v)}
                  options={[["side-on","Side-on"],["square","Square"],["open","Open"]]} />
                <ButtonGrid label="Stance width" value={data.width} onChange={(v) => set("width", v)}
                  options={[["narrow","Narrow"],["shoulder","Shoulder"],["wide","Wide"]]} />
                <ButtonGrid label="Guard" value={data.guard} onChange={(v) => set("guard", v)}
                  options={[["leg","Leg"],["middle","Middle"],["off","Off"]]} />
              </Group>
            )}
            {step === 3 && (
              <Group title="Grip & backlift">
                <ButtonGrid label="Grip type" value={data.grip} onChange={(v) => set("grip", v)}
                  options={[["v","V-grip"],["o","O-grip"],["bottom","Bottom-hand"]]} />
                <ButtonGrid label="Backlift direction" value={data.backlift} onChange={(v) => set("backlift", v)}
                  options={[["straight","Straight up"],["second-slip","Toward 2nd slip"],["gully","Toward gully"]]} />
              </Group>
            )}
            {step === 4 && (
              <Group title="Review & analyze">
                <div className="rounded-xl border border-border divide-y divide-border">
                  {[
                    ["Video", file ? file.name : "No file selected"],
                    ["Batting hand", data.hand],
                    ["Level", data.level],
                    ["Foot position", data.foot],
                    ["Stance width", data.width],
                    ["Guard", data.guard],
                    ["Grip", data.grip],
                    ["Backlift", data.backlift],
                  ].map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between px-4 py-2.5 text-sm">
                      <span className="text-muted-foreground">{k}</span>
                      <span className="font-medium capitalize">{String(v).replace("-", " ")}</span>
                    </div>
                  ))}
                </div>
              </Group>
            )}
          </div>

          <div className="mt-6 flex items-center justify-between gap-3">
            <Button variant="ghost" disabled={step === 1} onClick={() => setStep(step - 1)}>
              Back
            </Button>
            {step < 4 ? (
              <Button onClick={() => setStep(step + 1)} className="gap-1.5">
                Continue <ChevronRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={() => navigate({ to: "/loading" })}
                className="h-12 px-6 text-base gap-2 font-semibold"
              >
                <Sparkles className="h-4 w-4" /> Analyze Stance
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stepper({ step }) {
  return (
    <div className="flex items-center gap-2">
      {STEPS.map((s, i) => {
        const done = step > s.id;
        const active = step === s.id;
        return (
          <div key={s.id} className="flex items-center gap-2 flex-1">
            <div className={
              "h-7 w-7 shrink-0 rounded-full grid place-items-center text-xs font-semibold border " +
              (done ? "bg-primary text-primary-foreground border-primary"
                : active ? "bg-primary/15 text-primary border-primary/40"
                : "bg-muted text-muted-foreground border-border")
            }>
              {done ? <Check className="h-3.5 w-3.5" /> : s.id}
            </div>
            <div className={
              "text-xs font-medium " + (active || done ? "text-foreground" : "text-muted-foreground")
            }>
              {s.label}
            </div>
            {i < STEPS.length - 1 && <div className="flex-1 h-px bg-border" />}
          </div>
        );
      })}
    </div>
  );
}

function Group({ title, children }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground mb-3">{title}</div>
      <div className="space-y-5">{children}</div>
    </div>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <div>
      <div className="text-sm font-medium mb-2">{label}</div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-11 bg-input/40">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map(([v, l]) => <SelectItem key={v} value={v}>{l}</SelectItem>)}
        </SelectContent>
      </Select>
    </div>
  );
}

function ButtonGrid({ label, value, onChange, options }) {
  return (
    <div>
      <div className="text-sm font-medium mb-2">{label}</div>
      <div className="grid grid-cols-3 gap-2">
        {options.map(([v, l]) => {
          const sel = v === value;
          return (
            <button
              key={v}
              type="button"
              onClick={() => onChange(v)}
              className={
                "h-11 rounded-xl text-sm font-medium border transition-colors " +
                (sel
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-muted/40 text-foreground border-border hover:bg-muted")
              }
            >
              {l}
            </button>
          );
        })}
      </div>
    </div>
  );
}
