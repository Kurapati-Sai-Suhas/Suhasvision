import React from "react";
import { useNavigate, useRouterState, Link } from "@tanstack/react-router";
import {
  Activity,
  Bell,
  Command,
  Flame,
  LayoutGrid,
  LogOut,
  Search,
  Settings,
  Sparkles,
  Users,
  Video,
  Trophy,
  Upload,
  LineChart,
  Inbox,
  UserCheck,
  Search as SearchIcon,
} from "lucide-react";
// import { useAuth } from "@/context/AuthContext";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

// `badge` counts are NOT set here -- they used to be hardcoded literals
// (queue always "12", feedback always "1") shown regardless of real data.
// Real counts are passed in via the `navBadges` prop by each dashboard,
// which has access to the actual queue/inbox length.
const COACH_NAV = [
  { key: "overview", label: "Command Center", icon: LayoutGrid },
  { key: "queue", label: "Review Queue", icon: Video },
  { key: "rankings", label: "Player Rankings", icon: Trophy },
  { key: "talent", label: "Talent Scouting", icon: UserCheck },
  { key: "analysis", label: "Video Analysis", icon: Activity },
];

const LEARNER_NAV = [
  { key: "overview", label: "My Cricket", icon: LayoutGrid },
  { key: "upload", label: "Upload Center", icon: Upload },
  { key: "progress", label: "Progress Timeline", icon: LineChart },
  { key: "profile", label: "Gamification", icon: Sparkles },
  { key: "feedback", label: "Feedback Inbox", icon: Inbox },
];

export const BrandMark = ({ size = "md" }) => (
  <div className="flex items-center gap-2.5">
    <div
      className={`relative inline-flex items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400/90 via-emerald-500 to-emerald-700 shadow-[0_0_24px_-6px_rgba(16,185,129,0.7)] ${
        size === "lg" ? "h-11 w-11" : "h-9 w-9"
      }`}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className={size === "lg" ? "h-6 w-6" : "h-5 w-5"}
        strokeWidth="2"
        stroke="#05080F"
      >
        <path d="M4 12c0-4.4 3.6-8 8-8" strokeLinecap="round" />
        <path d="M12 4c4.4 0 8 3.6 8 8" strokeLinecap="round" />
        <circle cx="12" cy="12" r="2" fill="#05080F" stroke="none" />
        <path d="M12 14v6" strokeLinecap="round" />
      </svg>
      <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-[#D4AF37] shadow-[0_0_10px_2px_rgba(212,175,55,0.6)]" />
    </div>
    <div className="leading-none">
      <div className={`font-display font-bold tracking-tight text-slate-50 ${size === "lg" ? "text-xl" : "text-lg"}`}>
        SuhasVision
      </div>
      <div className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.24em] text-slate-500">
        Cricket · Biomechanics · AI
      </div>
    </div>
  </div>
);

const AppShell = ({ children, activeKey, onNavigate, role, title, subtitle, rightSlot, profile = {}, navBadges = {}, pulse = null }) => {
  const session = { name: profile.name || "User", role: role };
  const logout = () => { localStorage.clear(); };
  const navigate = useNavigate();
  const location = useRouterState({ select: (s) => s.location });

  const nav = role === "coach" ? COACH_NAV : LEARNER_NAV;

  const handleLogout = () => {
    logout();
    navigate({ to: "/", replace: true });
  };

  return (
    <div className="relative flex min-h-screen w-full bg-[#05080F] text-slate-100">
      {/* Ambient background */}
      <div className="pointer-events-none fixed inset-0 sv-ambient" aria-hidden />
      <div className="pointer-events-none fixed inset-0 sv-grid opacity-60" aria-hidden />

      {/* Sidebar */}
      <aside
        data-testid="app-sidebar"
        className="relative z-10 hidden w-[264px] shrink-0 flex-col border-r border-white/5 bg-[#070B15]/80 px-4 py-6 backdrop-blur-xl lg:flex"
      >
        <button
          data-testid="brand-home"
          onClick={() => navigate({ to: role === "coach" ? "/coach" : "/learner" })}
          className="mb-8 flex items-center gap-2 px-2 text-left"
        >
          <BrandMark />
        </button>

        <div className="mb-3 px-2 text-[10px] font-semibold uppercase tracking-[0.28em] text-slate-500">
          {role === "coach" ? "Coach Console" : "Player Workspace"}
        </div>

        <nav className="flex flex-col gap-1">
          {nav.map((item) => {
            const Icon = item.icon;
            const isActive = activeKey === item.key;
            return (
              <button
                key={item.key}
                data-testid={`nav-${item.key}`}
                onClick={() => onNavigate(item.key)}
                className={`group flex items-center justify-between rounded-xl px-3 py-2.5 text-sm transition-all duration-300 ${
                  isActive
                    ? "bg-gradient-to-r from-emerald-500/15 via-emerald-500/10 to-transparent text-emerald-100 shadow-[inset_0_0_0_1px_rgba(16,185,129,0.25)]"
                    : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
                }`}
              >
                <span className="flex items-center gap-3">
                  <Icon
                    className={`h-4 w-4 ${
                      isActive ? "text-emerald-300" : "text-slate-500 group-hover:text-slate-300"
                    }`}
                  />
                  <span className="font-medium tracking-tight">{item.label}</span>
                </span>
                {navBadges[item.key] ? (
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                      isActive
                        ? "bg-emerald-500/25 text-emerald-100"
                        : "bg-white/5 text-slate-400 group-hover:bg-white/10"
                    }`}
                  >
                    {navBadges[item.key]}
                  </span>
                ) : null}
              </button>
            );
          })}
        </nav>

        <div className="mt-auto flex flex-col gap-3">
          <div className="rounded-2xl border border-white/5 bg-gradient-to-br from-[#0C1322] to-[#0A1020] p-4">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-emerald-300">
              <Flame className="h-3.5 w-3.5" />
              {pulse?.label || (role === "coach" ? "Academy Pulse" : "Daily Streak")}
            </div>
            <div className="mt-3 font-display text-3xl font-bold text-slate-50">
              {pulse?.value ?? (role === "coach" ? "—" : `${profile.streak || 0}`)}
              <span className="ml-1 text-sm font-medium text-slate-500">{pulse?.unit ?? (role === "coach" ? "" : "days")}</span>
            </div>
            <div className="mt-1 text-xs text-slate-400">
              {pulse?.sub || (role === "coach" ? "No sessions reviewed yet" : "Log a session to build your streak")}
            </div>
          </div>
          <button
            data-testid="logout-btn"
            onClick={handleLogout}
            className="flex items-center gap-2 rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 text-sm text-slate-400 transition-all hover:border-white/10 hover:bg-white/5 hover:text-slate-100"
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main column */}
      <main className="relative z-10 flex min-h-screen flex-1 flex-col">
        {/* Topbar */}
        <header
          data-testid="app-topbar"
          className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-white/5 bg-[#05080F]/70 px-5 backdrop-blur-xl md:px-8"
        >
          <div className="flex items-center gap-4">
            <div className="lg:hidden">
              <BrandMark />
            </div>
            <div className="hidden lg:block">
              <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-500">
                {role === "coach" ? "Coach" : "Player"} · {profile.name}
              </div>
              {title ? (
                <div className="font-display text-lg font-semibold tracking-tight text-slate-50">
                  {title}
                </div>
              ) : null}
            </div>
          </div>
          <div className="flex items-center gap-2 md:gap-3">
            <div className="hidden items-center gap-2 rounded-full border border-white/5 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-400 md:flex">
              <Search className="h-3.5 w-3.5" />
              <span>Search players, drills, sessions…</span>
              <span className="ml-2 rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-slate-500">
                ⌘K
              </span>
            </div>
            {/* No notifications feature/endpoint exists yet -- this used to
                unconditionally show a "new notification" dot on every page
                load with no backing data and no onClick. Removed the fake
                indicator rather than implying an unread notification that
                doesn't exist; re-add it once there's a real source to read
                from. */}
            <button
              data-testid="notifications-btn"
              className="relative flex h-9 w-9 items-center justify-center rounded-full border border-white/5 bg-white/[0.03] text-slate-300 transition-all hover:border-white/10 hover:bg-white/5"
            >
              <Bell className="h-4 w-4" />
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  data-testid="profile-menu"
                  className="flex items-center gap-2 rounded-full border border-white/5 bg-white/[0.03] py-1 pl-1 pr-3 text-sm transition-all hover:border-white/10 hover:bg-white/5"
                >
                  <Avatar className="h-7 w-7">
                    <AvatarImage src={profile.avatar} alt={profile.name} />
                    <AvatarFallback className="bg-emerald-500/20 text-xs text-emerald-200">
                      {profile.name
                        .split(" ")
                        .map((s) => s[0])
                        .slice(0, 2)
                        .join("")}
                    </AvatarFallback>
                  </Avatar>
                  <span className="hidden text-slate-200 md:inline">{session?.name || profile.name}</span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                className="w-56 border-white/5 bg-[#0C1322]/95 text-slate-200 backdrop-blur-xl"
              >
                <DropdownMenuLabel>
                  <div className="text-sm font-medium text-slate-100">{profile.name}</div>
                  <div className="text-xs text-slate-500">{profile.email}</div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-white/5" />
                <DropdownMenuItem
                  data-testid="menu-profile"
                  className="focus:bg-white/5 focus:text-slate-100"
                >
                  <UserCheck className="mr-2 h-4 w-4" />
                  Profile & Preferences
                </DropdownMenuItem>
                <DropdownMenuItem
                  data-testid="menu-settings"
                  className="focus:bg-white/5 focus:text-slate-100"
                >
                  <Settings className="mr-2 h-4 w-4" />
                  Academy settings
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-white/5" />
                <DropdownMenuItem
                  data-testid="menu-logout"
                  onClick={handleLogout}
                  className="text-rose-300 focus:bg-rose-500/10 focus:text-rose-200"
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        {/* Page content */}
        <div className="relative z-10 flex-1 px-5 py-8 md:px-10 md:py-12">{children}</div>
      </main>
    </div>
  );
};

export default AppShell;
