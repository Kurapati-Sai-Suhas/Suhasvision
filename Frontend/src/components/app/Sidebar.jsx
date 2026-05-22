import { Link, useRouterState, useNavigate } from "@tanstack/react-router";
import { LayoutDashboard, Upload, GitCompareArrows, LogOut, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyzer", label: "New Analysis", icon: Upload },
  { to: "/compare", label: "Compare", icon: GitCompareArrows },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const path = useRouterState({ select: (s) => s.location.pathname });

  return (
    <aside className="hidden md:flex flex-col w-20 lg:w-60 shrink-0 border-r border-border bg-sidebar text-sidebar-foreground h-screen sticky top-0">
      <div className="flex items-center gap-2 px-5 py-6">
        <div className="h-9 w-9 rounded-lg bg-primary text-primary-foreground grid place-items-center">
          <Activity className="h-5 w-5" />
        </div>
        <div className="hidden lg:block">
          <div className="text-sm font-semibold tracking-tight">SuhasVision</div>
          <div className="text-[11px] text-muted-foreground">Cricket analytics</div>
        </div>
      </div>

      <nav className="flex-1 px-3 space-y-1">
        {items.map(({ to, label, icon: Icon }) => {
          const active = path.startsWith(to);
          return (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground"
              )}
            >
              <Icon className="h-[18px] w-[18px] shrink-0" />
              <span className="hidden lg:inline">{label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-3 border-t border-sidebar-border">
        <button
          onClick={() => navigate({ to: "/" })}
          className="w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
        >
          <LogOut className="h-[18px] w-[18px]" />
          <span className="hidden lg:inline">Logout</span>
        </button>
      </div>
    </aside>
  );
}
