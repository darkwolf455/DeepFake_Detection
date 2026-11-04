
import React from 'react';
import {
  ShieldCheck,
  ScanSearch,
  History,
  BookOpen,
  Code2,
  Settings,
  Sun,
  Moon,
  ChevronRight,
  PanelLeft,
  Fingerprint,
} from 'lucide-react';

import { useTheme } from '../context/ThemeContext';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  hasActiveResult: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  hasActiveResult,
}) => {
  const { theme, toggleTheme } = useTheme();

  const primaryItems = [
    { id: 'analyze', label: 'Analyze', description: 'Inspect media', icon: ScanSearch },
    ...(hasActiveResult
      ? [{ id: 'results', label: 'Results', description: 'Latest examination', icon: ShieldCheck }]
      : []),
    { id: 'history', label: 'History', description: 'Past investigations', icon: History },
  ];

  const intelligenceItems = [
    { id: 'methodology', label: 'Methodology', description: 'How detection works', icon: BookOpen },
    { id: 'code', label: 'Code Explorer', description: 'Explore implementation', icon: Code2 },
  ];

  const settingsItem = {
    id: 'settings',
    label: 'Settings',
    description: 'Configure workspace',
    icon: Settings,
  };

  const renderNavItem = (item: (typeof primaryItems)[number]) => {
    const Icon = item.icon;
    const isActive = activeTab === item.id;

    return (
      <button
        key={item.id}
        type="button"
        onClick={() => setActiveTab(item.id)}
        aria-current={isActive ? 'page' : undefined}
        className={`group relative flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-all duration-200 ${
          isActive
            ? 'bg-[var(--accent-soft)] text-[var(--accent)]'
            : 'text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]'
        }`}
      >
        {isActive && (
          <span className="absolute bottom-2 left-0 top-2 w-[3px] rounded-r-full bg-[var(--accent)]" />
        )}

        <span
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition-colors ${
            isActive
              ? 'bg-[var(--accent)] text-white shadow-sm'
              : 'bg-[var(--surface-2)] text-[var(--muted)] group-hover:text-[var(--text)]'
          }`}
        >
          <Icon size={17} strokeWidth={1.9} />
        </span>

        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-bold">
            {item.label}
          </span>
          <span
            className={`mt-0.5 block truncate text-[10px] ${
              isActive ? 'text-[var(--accent)]/80' : 'text-[var(--muted)]'
            }`}
          >
            {item.description}
          </span>
        </span>

        {isActive && (
          <ChevronRight size={15} className="shrink-0 opacity-70" />
        )}
      </button>
    );
  };

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[264px] flex-col border-r border-[var(--border)] bg-[var(--surface)] lg:flex">
        {/* Brand */}
        <button
          type="button"
          onClick={() => setActiveTab('analyze')}
          className="flex min-h-[86px] items-center gap-3 border-b border-[var(--border)] px-6 text-left"
          aria-label="VeriSense home"
        >
          <div className="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-[15px] bg-gradient-to-br from-[#f18b43] via-[#a16ce8] to-[#5c8be8] shadow-lg shadow-violet-500/15">
            <ShieldCheck className="h-6 w-6 text-white" strokeWidth={1.8} />
            <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-[3px] border-[var(--surface)] bg-[var(--success)]" />
          </div>

          <span className="min-w-0">
            <span className="flex items-center gap-2">
              <span className="text-[21px] font-black tracking-[-0.06em] text-[var(--text)]">
                VeriSense
              </span>
              <span className="rounded-md border border-[var(--accent-border)] bg-[var(--accent-soft)] px-1.5 py-0.5 text-[9px] font-extrabold tracking-wide text-[var(--accent)]">
                V1.0
              </span>
            </span>
            <span className="mt-0.5 block text-[9px] font-bold tracking-[0.16em] text-[var(--muted)]">
              MULTIMODAL FORENSICS
            </span>
          </span>
        </button>

        {/* Workspace identity */}
        <div className="mx-4 mt-5 rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)]">
              <Fingerprint size={17} />
            </div>
            <div className="min-w-0">
              <p className="truncate text-[11px] font-bold text-[var(--text)]">
                Forensic workspace
              </p>
              <p className="mt-0.5 text-[9px] text-[var(--muted)]">
                Digital media intelligence
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-4 pt-7" aria-label="Main navigation">
          <p className="mb-3 px-3 text-[9px] font-extrabold uppercase tracking-[0.2em] text-[var(--muted)]">
            Workspace
          </p>

          <div className="space-y-1">
            {primaryItems.map(renderNavItem)}
          </div>

          <p className="mb-3 mt-8 px-3 text-[9px] font-extrabold uppercase tracking-[0.2em] text-[var(--muted)]">
            Intelligence
          </p>

          <div className="space-y-1">
            {intelligenceItems.map(renderNavItem)}
          </div>
        </nav>

        {/* Bottom controls */}
        <div className="border-t border-[var(--border)] p-4">
          {renderNavItem(settingsItem)}

          <div className="mt-3 flex items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-3">
            <div className="flex items-center gap-2.5">
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--success)] opacity-40" />
                <span className="relative h-2.5 w-2.5 rounded-full bg-[var(--success)]" />
              </span>
              <span>
                <span className="block text-[10px] font-bold text-[var(--text)]">
                  Workspace active
                </span>
                <span className="mt-0.5 block text-[9px] text-[var(--muted)]">
                  VeriSense V1.0
                </span>
              </span>
            </div>

            <button
              type="button"
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] transition hover:border-[var(--accent)] hover:bg-[var(--accent-soft)]"
            >
              {theme === 'dark' ? (
                <Sun size={17} className="text-amber-400" />
              ) : (
                <Moon size={17} className="text-indigo-500" />
              )}
            </button>
          </div>

          <p className="mt-4 text-center text-[9px] tracking-wide text-[var(--muted)]">
            © 2026 VERISENSE · DIGITAL FORENSICS
          </p>
        </div>
      </aside>

      {/* Mobile / tablet navigation */}
      <div className="sticky top-0 z-50 w-full border-b border-[var(--border)] bg-[var(--surface)] lg:hidden">
        <div className="flex min-h-[64px] items-center justify-between gap-3 px-4">
          <button
            type="button"
            onClick={() => setActiveTab('analyze')}
            className="flex items-center gap-2.5"
            aria-label="VeriSense home"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-[#f18b43] via-[#a16ce8] to-[#5c8be8]">
              <ShieldCheck className="h-5 w-5 text-white" />
            </span>
            <span className="text-lg font-black tracking-tight text-[var(--text)]">
              VeriSense
            </span>
          </button>

          <div className="flex items-center gap-2">
            <span className="hidden items-center gap-1.5 rounded-full border border-[var(--border)] px-2.5 py-1.5 text-[9px] font-bold text-[var(--text-secondary)] sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--success)]" />
              ACTIVE
            </span>
            <button
              type="button"
              onClick={toggleTheme}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface-2)]"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? (
                <Sun size={17} className="text-amber-400" />
              ) : (
                <Moon size={17} className="text-indigo-500" />
              )}
            </button>
          </div>
        </div>

        <nav
          className="flex gap-1 overflow-x-auto px-3 pb-3"
          aria-label="Mobile navigation"
        >
          {[...primaryItems, ...intelligenceItems, settingsItem].map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-xs font-bold transition ${
                  isActive
                    ? 'bg-[var(--accent-soft)] text-[var(--accent)]'
                    : 'text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]'
                }`}
              >
                <Icon size={14} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </>
  );
};
