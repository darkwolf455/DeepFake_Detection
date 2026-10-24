import React, { useState, useEffect } from 'react';
import {
  Settings,
  Server,
  Activity,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Sun,
  Moon,
  RefreshCw,
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
  apiBaseUrl,
  setApiBaseUrl,
  inconclusiveLower,
  setInconclusiveLower,
  inconclusiveUpper,
  setInconclusiveUpper,
}) => {
  const { theme, setTheme } = useTheme();
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const fetchHealth = async () => {
    setIsCheckingHealth(true);
    const data = await checkSystemHealth(apiBaseUrl);
    setHealthStatus(data);
    setIsCheckingHealth(false);
  };

  useEffect(() => {
    fetchHealth();
  }, [apiBaseUrl]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 sm:py-12 space-y-8">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white flex items-center gap-2.5">
          <Settings className="w-7 h-7 text-cyan-400" />
          <span>Platform Settings & Health</span>
        </h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1">
          Configure API endpoints, threshold decision bands, visual theme, and check backend neural hardware telemetry.
        </p>
      </div>

      {/* Backend API Server Configuration */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>Forensic Backend API Endpoint</span>
          </div>
          <button
            onClick={fetchHealth}
            disabled={isCheckingHealth}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isCheckingHealth ? 'animate-spin text-cyan-400' : ''}`} />
            <span>Test Connection</span>
          </button>
        </div>

        <p className="text-xs text-slate-400">
          Specify the base URL for the forensic backend. Leave blank to use the built-in server, or point to your local FastAPI server running in VS Code (e.g. <code className="text-cyan-400">http://localhost:8000</code>).
        </p>

        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="http://localhost:8000 (or leave empty for built-in)"
            value={apiBaseUrl}
            onChange={(e) => setApiBaseUrl(e.target.value)}
            className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
          />
          <button
            onClick={() => setApiBaseUrl('')}
            className="px-3 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors"
          >
            Reset Default
          </button>
        </div>

        {/* Live Health Status Card */}
        {healthStatus && (
          <div className="mt-4 p-4 rounded-2xl bg-slate-950 border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
            <div className="flex items-center gap-2">
              {healthStatus.status === 'ok' ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertCircle className="w-4 h-4 text-rose-400" />
              )}
              <span className="text-slate-300">
                Server Status: <strong className={healthStatus.status === 'ok' ? 'text-emerald-400' : 'text-rose-400'}>{healthStatus.status.toUpperCase()}</strong>
              </span>
            </div>
            <div className="flex items-center gap-4 text-slate-400 text-[11px]">
              <span>Device: <strong className="text-cyan-400">{healthStatus.device}</strong></span>
              <span>VeriSense v{healthStatus.version}</span>
            </div>
          </div>
        )}
      </div>

      {/* Decision Thresholds (Inconclusive Band) */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center gap-2 text-white font-bold text-sm">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span>Forensic Inconclusive Margin Calibration</span>
        </div>
        <p className="text-xs text-slate-400">
          Media with calibrated probabilities within this interval are designated as <strong>inconclusive</strong> to prevent false accusations caused by lossy social compression or mixed generation.
        </p>

        <div className="space-y-4 pt-2">
          <div>
            <div className="flex justify-between text-xs font-mono mb-1 text-slate-300">
              <span>Authentic Threshold (Upper bound for Authentic):</span>
              <span className="text-cyan-400 font-bold">{Math.round(inconclusiveLower * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.20"
              max="0.49"
              step="0.01"
              value={inconclusiveLower}
              onChange={(e) => setInconclusiveLower(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1 text-slate-300">
              <span>Manipulated Threshold (Lower bound for Deepfake):</span>
              <span className="text-rose-400 font-bold">{Math.round(inconclusiveUpper * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.51"
              max="0.85"
              step="0.01"
              value={inconclusiveUpper}
              onChange={(e) => setInconclusiveUpper(parseFloat(e.target.value))}
              className="w-full accent-rose-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-950 text-[11px] font-mono text-slate-400">
          Active Decision Rule: &lt;{Math.round(inconclusiveLower * 100)}% = Authentic • {Math.round(inconclusiveLower * 100)}%–{Math.round(inconclusiveUpper * 100)}% = Inconclusive • &gt;{Math.round(inconclusiveUpper * 100)}% = Manipulated
        </div>
      </div>

      {/* Theme Selection */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl flex items-center justify-between">
        <div>
          <h4 className="text-sm font-bold text-white">Visual Appearance</h4>
          <p className="text-xs text-slate-400 mt-0.5">Toggle dark-first forensic interface or daylight mode.</p>
        </div>
        <div className="flex items-center gap-2 p-1 bg-slate-950 rounded-xl border border-slate-800">
          <button
            onClick={() => setTheme('dark')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              theme === 'dark' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400'
            }`}
          >
            <Moon className="w-3.5 h-3.5" />
            <span>Dark</span>
          </button>
          <button
            onClick={() => setTheme('light')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              theme === 'light' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'text-slate-400'
            }`}
          >
            <Sun className="w-3.5 h-3.5" />
            <span>Light</span>
          </button>
        </div>
      </div>
    </div>
  );
};
