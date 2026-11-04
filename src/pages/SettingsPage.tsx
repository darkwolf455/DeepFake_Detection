
import React, { useState, useEffect } from 'react';
import {
  Settings, Server, Activity, CheckCircle2, AlertCircle,
  Sliders, Sun, Moon, RefreshCw, ShieldCheck, Gauge
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { checkSystemHealth } from '../services/api';

interface SettingsPageProps {
  apiBaseUrl: string;
  setApiBaseUrl: (url: string) => void;
  inconclusiveLower: number;
  setInconclusiveLower: (val: number) => void;
  inconclusiveUpper: number;
  setInconclusiveUpper: (val: number) => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  apiBaseUrl, setApiBaseUrl, inconclusiveLower,
  setInconclusiveLower, inconclusiveUpper, setInconclusiveUpper
}) => {
  const { theme, setTheme } = useTheme();
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const fetchHealth = async () => {
    setIsCheckingHealth(true);
    try {
      const data = await checkSystemHealth(apiBaseUrl);
      setHealthStatus(data);
    } catch {
      setHealthStatus({ status: 'error', device: 'Unavailable', version: '—' });
    } finally {
      setIsCheckingHealth(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, [apiBaseUrl]);

  return (
    <main className="vs-page min-h-screen px-4 py-9 sm:px-8 lg:px-12">
      <div className="max-w-5xl mx-auto space-y-8">

        <section>
          <div className="flex items-center gap-2 text-cyan-500 text-[11px] font-bold uppercase tracking-[0.2em] mb-3">
            <Activity size={15} /> System Control / Configuration
          </div>
          <h1 className="vs-heading text-3xl sm:text-4xl font-bold">
            Platform Settings<span className="text-cyan-500">.</span>
          </h1>
          <p className="vs-muted text-sm mt-2">
            Configure forensic infrastructure, detection thresholds and interface preferences.
          </p>
        </section>

        <section className="vs-card rounded-2xl border p-5 sm:p-7 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl bg-cyan-500/10 flex items-center justify-center text-cyan-500">
                <Server size={21} />
              </div>
              <div>
                <h3 className="vs-heading font-bold text-sm">Backend Infrastructure</h3>
                <p className="vs-muted text-xs mt-1">API endpoint and connection health</p>
              </div>
            </div>
            <button
              onClick={fetchHealth}
              disabled={isCheckingHealth}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-500 text-slate-950 font-bold text-xs hover:bg-cyan-400 disabled:opacity-50"
            >
              <RefreshCw size={14} className={isCheckingHealth ? 'animate-spin' : ''} />
              Test connection
            </button>
          </div>

          <div>
            <label className="vs-muted text-xs block mb-2">Backend API Base URL</label>
            <input
              value={apiBaseUrl}
              onChange={e => setApiBaseUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="vs-input w-full rounded-xl px-4 py-3 text-xs font-mono outline-none"
            />
            <p className="vs-muted text-[11px] mt-2">
              Leave empty to use the built-in server.
            </p>
          </div>

          <button
            onClick={() => setApiBaseUrl('')}
            className="text-xs text-cyan-500 hover:underline"
          >
            Reset to default endpoint
          </button>

          {healthStatus && (
            <div className="rounded-xl border vs-panel p-4 flex flex-wrap justify-between items-center gap-3">
              <div className="flex items-center gap-2">
                {healthStatus.status === 'ok'
                  ? <CheckCircle2 className="text-emerald-500" size={19} />
                  : <AlertCircle className="text-rose-500" size={19} />}
                <div>
                  <div className="vs-heading text-xs font-bold">Server {healthStatus.status === 'ok' ? 'Operational' : 'Unavailable'}</div>
                  <div className="vs-muted text-[10px] mt-1">Connection health check</div>
                </div>
              </div>
              <div className="vs-muted text-xs font-mono">
                {healthStatus.device} / v{healthStatus.version}
              </div>
            </div>
          )}
        </section>

        <section className="vs-card rounded-2xl border p-5 sm:p-7 space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-violet-500/10 flex items-center justify-center text-violet-500">
              <Gauge size={21} />
            </div>
            <div>
              <h3 className="vs-heading font-bold text-sm">Detection Calibration</h3>
              <p className="vs-muted text-xs mt-1">Configure the inconclusive decision band</p>
            </div>
          </div>

          <p className="vs-muted text-xs leading-relaxed">
            Results inside the selected interval are marked inconclusive to avoid overconfident classifications.
          </p>

          <div className="space-y-6">
            <div>
              <div className="flex justify-between text-xs mb-3">
                <label className="vs-heading font-semibold">Authentic upper threshold</label>
                <strong className="text-cyan-500">{Math.round(inconclusiveLower * 100)}%</strong>
              </div>
              <input
                type="range" min="0.20" max="0.49" step="0.01"
                value={inconclusiveLower}
                onChange={e => setInconclusiveLower(Math.min(parseFloat(e.target.value), inconclusiveUpper - 0.02))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-3">
                <label className="vs-heading font-semibold">Manipulated lower threshold</label>
                <strong className="text-rose-500">{Math.round(inconclusiveUpper * 100)}%</strong>
              </div>
              <input
                type="range" min="0.51" max="0.85" step="0.01"
                value={inconclusiveUpper}
                onChange={e => setInconclusiveUpper(Math.max(parseFloat(e.target.value), inconclusiveLower + 0.02))}
                className="w-full accent-rose-500 cursor-pointer"
              />
            </div>
          </div>

          <div className="vs-panel rounded-xl p-4">
            <div className="flex items-center gap-2 text-cyan-500 text-xs font-bold mb-3">
              <ShieldCheck size={15} /> Active Decision Rule
            </div>
            <div className="flex h-2 overflow-hidden rounded-full mb-3">
              <div className="bg-emerald-500" style={{ width: `${inconclusiveLower * 100}%` }} />
              <div className="bg-amber-500" style={{ width: `${(inconclusiveUpper - inconclusiveLower) * 100}%` }} />
              <div className="bg-rose-500 flex-1" />
            </div>
            <div className="grid grid-cols-3 gap-2 text-[10px]">
              <div className="text-emerald-500">Authentic<br/><strong>Below {Math.round(inconclusiveLower * 100)}%</strong></div>
              <div className="text-amber-500 text-center">Inconclusive<br/><strong>{Math.round(inconclusiveLower * 100)}–{Math.round(inconclusiveUpper * 100)}%</strong></div>
              <div className="text-rose-500 text-right">Manipulated<br/><strong>Above {Math.round(inconclusiveUpper * 100)}%</strong></div>
            </div>
          </div>
        </section>

        <section className="vs-card rounded-2xl border p-5 sm:p-7">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-11 h-11 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-500">
              <Settings size={21} />
            </div>
            <div>
              <h3 className="vs-heading font-bold text-sm">Interface Preferences</h3>
              <p className="vs-muted text-xs mt-1">Choose your workspace appearance</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setTheme('dark')}
              className={`rounded-xl border p-4 text-left transition-all ${
                theme === 'dark' ? 'border-cyan-500 bg-cyan-500/10' : 'vs-panel'
              }`}
            >
              <Moon size={20} className="text-cyan-500 mb-3" />
              <div className="vs-heading font-bold text-sm">Dark Mode</div>
              <div className="vs-muted text-[11px] mt-1">Midnight forensic workspace</div>
              {theme === 'dark' && <span className="text-cyan-500 text-[10px] mt-3 block">● ACTIVE</span>}
            </button>

            <button
              onClick={() => setTheme('light')}
              className={`rounded-xl border p-4 text-left transition-all ${
                theme === 'light' ? 'border-cyan-500 bg-cyan-500/10' : 'vs-panel'
              }`}
            >
              <Sun size={20} className="text-amber-500 mb-3" />
              <div className="vs-heading font-bold text-sm">Light Mode</div>
              <div className="vs-muted text-[11px] mt-1">Clean daylight workspace</div>
              {theme === 'light' && <span className="text-cyan-500 text-[10px] mt-3 block">● ACTIVE</span>}
            </button>
          </div>
        </section>

      </div>
    </main>
  );
};
