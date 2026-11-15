import {
  BookOpen,
  Boxes,
  History,
  Moon,
  ScanLine,
  Settings2,
  ShieldCheck,
  Sun,
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export function Navbar({
  activeTab,
  setActiveTab,
  hasActiveResult,
}: {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  hasActiveResult: boolean;
}) {
  const { theme, toggleTheme } = useTheme();
  const items = [
    { id: 'analyze', label: 'Analyze media', icon: ScanLine },
    ...(hasActiveResult
      ? [{ id: 'results', label: 'Latest result', icon: ShieldCheck }]
      : []),
    { id: 'history', label: 'History', icon: History },
    { id: 'models', label: 'Model registry', icon: Boxes },
    { id: 'methodology', label: 'Methodology', icon: BookOpen },
    { id: 'settings', label: 'Settings', icon: Settings2 },
  ];
  const themeButton = (
    <button
      className="icon-button"
      onClick={toggleTheme}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? <Sun /> : <Moon />}
    </button>
  );
  return (
    <aside className="sidebar no-print">
      <div className="brand-row">
        <button
          className="brand"
          onClick={() => setActiveTab('analyze')}
          aria-label="VeriSense home"
        >
          <span className="brand-mark">
            <ShieldCheck size={23} strokeWidth={1.6} />
          </span>
          <span>
            <span className="brand-name">
              VeriSense<span style={{ color: 'var(--accent)' }}>.</span>
            </span>
            <span className="brand-caption">MEDIA FORENSICS LAB</span>
          </span>
        </button>
        {themeButton}
      </div>
      <nav aria-label="Main navigation">
        <div className="nav-label">Your workspace</div>
        <div className="nav-items">
          {items.map(({ id, label, icon: Icon }, index) => (
            <button
              key={id}
              className={`nav-button ${activeTab === id ? 'active' : ''}`}
              onClick={() => setActiveTab(id)}
              aria-current={activeTab === id ? 'page' : undefined}
            >
              <Icon strokeWidth={1.7} />
              <span>{label}</span>
              <span className="nav-number">0{index + 1}</span>
            </button>
          ))}
        </div>
      </nav>
      <div className="sidebar-foot">
        <div
          className="actions"
          style={{ justifyContent: 'space-between', marginBottom: 14 }}
        >
          <span className="tag">STUDENT EDITION</span>
          {themeButton}
        </div>
        <p>A closer look at digital media.</p>
        <span>Local inference. Measured limitations. No paid API.</span>
      </div>
    </aside>
  );
}
