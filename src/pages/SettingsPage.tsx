import { useEffect, useState } from 'react';
import {
  Check,
  Link2,
  Moon,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sun,
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { checkSystemHealth } from '../services/api';

export function SettingsPage({
  apiBaseUrl,
  setApiBaseUrl,
  health,
  checking,
  onCheck,
}: {
  apiBaseUrl: string;
  setApiBaseUrl: (url: string) => void;
  health: Awaited<ReturnType<typeof checkSystemHealth>> | null;
  checking: boolean;
  onCheck: () => void;
}) {
  const { theme, setTheme } = useTheme();
  const [draft, setDraft] = useState(apiBaseUrl);
  const [error, setError] = useState('');
  useEffect(() => setDraft(apiBaseUrl), [apiBaseUrl]);
  const save = () => {
    const value = draft.trim().replace(/\/$/, '');
    if (value) {
      try {
        const url = new URL(value);
        if (!['http:', 'https:'].includes(url.protocol)) throw new Error();
      } catch {
        setError(
          'Enter an http:// or https:// API address, or leave this empty for the local proxy.',
        );
        return;
      }
    }
    setError('');
    setApiBaseUrl(value);
  };
  return (
    <div className="page stack" style={{ maxWidth: 1000 }}>
      <section className="page-heading">
        <div className="eyebrow">
          <Settings2 size={13} /> Make it your workspace
        </div>
        <h1>A few settings. Nothing extra.</h1>
        <p>Connect your engine and choose how the workspace looks.</p>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2 className="section-title">
              <Link2 /> API connection
            </h2>
            <p>
              Saved in this browser. The default uses the local Python proxy.
            </p>
          </div>
          <span
            className={`tag ${health?.active_branches.length ? 'success' : ''}`}
          >
            {health?.active_branches.length ? 'CONNECTED' : 'OFFLINE'}
          </span>
        </div>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            save();
          }}
        >
          <label className="field-label" htmlFor="api-url">
            Backend API base URL
          </label>
          <input
            id="api-url"
            className="text-input font-mono"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Leave empty to use the local API proxy"
            spellCheck={false}
          />
          {error && (
            <p
              role="alert"
              className="small"
              style={{ color: 'var(--danger)', marginTop: 10 }}
            >
              {error}
            </p>
          )}
          <div className="actions" style={{ marginTop: 16 }}>
            <button className="primary-button" type="submit">
              <Check size={14} />
              Save endpoint
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={checking}
              onClick={onCheck}
            >
              <RefreshCw size={14} className={checking ? 'spinner' : ''} />
              Test saved connection
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                setDraft('');
                setApiBaseUrl('');
                setError('');
              }}
            >
              Use local proxy
            </button>
          </div>
        </form>
        <p className="small muted" style={{ marginTop: 18 }}>
          Device: {health?.device ?? '—'} · Loaded:{' '}
          {health?.active_branches.join(', ') || 'none reported'}
        </p>
      </section>
      <div className="two-column">
        <section className="panel">
          <h2 className="section-title">
            <Sun /> Appearance
          </h2>
          <p className="small" style={{ margin: '12px 0 20px' }}>
            Your theme follows you across every page.
          </p>
          <div className="actions">
            {(['dark', 'light'] as const).map((value) => (
              <button
                key={value}
                className={
                  theme === value ? 'primary-button' : 'secondary-button'
                }
                aria-pressed={theme === value}
                onClick={() => setTheme(value)}
              >
                {value === 'dark' ? <Moon size={15} /> : <Sun size={15} />}
                {value === 'dark' ? 'Midnight' : 'Daylight'}
              </button>
            ))}
          </div>
        </section>
        <section className="panel">
          <h2 className="section-title">
            <ShieldCheck /> Model decisions
          </h2>
          <p className="small" style={{ marginTop: 12 }}>
            Video, audio and fusion use cutoffs saved with the evaluated
            checkpoints. The image uncertainty band is configured on the
            backend.
          </p>
          <p className="small muted" style={{ marginTop: 12 }}>
            These settings do not change model weights or calibration.
          </p>
        </section>
      </div>
    </div>
  );
}
