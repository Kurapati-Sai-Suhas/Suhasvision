import { createFileRoute, Outlet } from "@tanstack/react-router";
import Sidebar from "@/components/app/Sidebar";

export const Route = createFileRoute("/_shell")({
  component: ShellLayout,
});

function ShellLayout() {
  return (
    <div className="flex min-h-screen w-full bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 min-w-0">
        <Outlet />
      </main>
    </div>
  );
}
