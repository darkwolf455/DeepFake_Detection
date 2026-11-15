import React, { useEffect, useState } from 'react';
import { ChevronRight, Fingerprint, RefreshCw } from 'lucide-react';
import { ThemeProvider } from './context/ThemeContext';
import { HistoryProvider, useHistory } from './context/HistoryContext';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { AnalyzePage } from './pages/AnalyzePage';
import { ResultsPage } from './pages/ResultsPage';
import { HistoryPage } from './pages/HistoryPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { ModelsPage } from './pages/ModelsPage';
import { SettingsPage } from './pages/SettingsPage';
import { AnalyzeResponse } from './types';
import { checkSystemHealth, DEFAULT_API_BASE_URL } from './services/api';

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: Error) {
    console.error('Workspace error:', error);
  }
  render() {
    return this.state.failed ? (
      <div className="page empty-state">
        <h1>The workspace needs a refresh.</h1>
        <p>Your saved analysis history remains in this browser.</p>
        <button
          className="primary-button"
          onClick={() => window.location.reload()}
        >
          Reload workspace
        </button>
      </div>
    ) : (
      this.props.children
    );
  }
}

function MainApp() {
  const [tab, setTab] = useState('analyze');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(() => {
    try {
      return localStorage.getItem('verisense_api_url') ?? DEFAULT_API_BASE_URL;
    } catch {
      return DEFAULT_API_BASE_URL;
    }
  });
  const [health, setHealth] = useState<Awaited<
    ReturnType<typeof checkSystemHealth>
  > | null>(null);
  const [checking, setChecking] = useState(false);
  const { addHistoryItem } = useHistory();
  const refreshHealth = async () => {
    setChecking(true);
    const next = await checkSystemHealth(apiBaseUrl);
    setHealth(next);
    setChecking(false);
  };
  useEffect(() => {
    let active = true;
    setChecking(true);
    setHealth(null);
    checkSystemHealth(apiBaseUrl).then((next) => {
      if (active) {
        setHealth(next);
        setChecking(false);
      }
    });
    try {
      localStorage.setItem('verisense_api_url', apiBaseUrl);
    } catch {
      /* Works for this session. */
    }
    return () => {
      active = false;
    };
  }, [apiBaseUrl]);
  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  const complete = (response: AnalyzeResponse, uploaded: File) => {
    setResult(response);
    setFile(uploaded);
    setPreview(URL.createObjectURL(uploaded));
    addHistoryItem({
      id: response.id,
      filename: uploaded.name,
      modality: response.modality,
      verdict: response.verdict,
      confidence: response.confidence,
      timestamp: new Date().toISOString(),
      scores: response.scores,
      file_sha256: response.file_sha256,
    });
    setTab('results');
  };
  const navigate = (next: string) => {
    setTab(next);
    window.scrollTo({ top: 0, behavior: 'instant' });
  };
  const titles: Record<string, string> = {
    analyze: 'Analyze media',
    results: 'Latest result',
    history: 'History',
    methodology: 'Methodology',
    models: 'Model registry',
    settings: 'Settings',
  };
  const ready = Boolean(health?.active_branches.length);

  return (
    <div className="app-shell">
      <a href="#workspace-main" className="sr-only focus:not-sr-only">
        Skip to content
      </a>
      <Navbar
        activeTab={tab}
        setActiveTab={navigate}
        hasActiveResult={Boolean(result)}
      />
      <div className="workspace">
        <header className="workspace-header">
          <div className="breadcrumb">
            <Fingerprint size={15} />
            <span>Workspace</span>
            <ChevronRight size={12} />
            <strong>{titles[tab]}</strong>
          </div>
          <div className="actions">
            <span className="session-label">LOCAL MEDIA FORENSICS</span>
            <span className={`tag ${ready ? 'success' : ''}`}>
              <span className="status-dot" />
              {checking
                ? 'Checking engine'
                : ready
                  ? `${health!.active_branches.length} models online`
                  : 'Engine offline'}
            </span>
            <button
              className="icon-button"
              onClick={refreshHealth}
              disabled={checking}
              aria-label="Refresh engine status"
            >
              <RefreshCw className={checking ? 'spinner' : ''} />
            </button>
          </div>
        </header>
        <main id="workspace-main" className="workspace-content">
          {tab === 'analyze' && (
            <AnalyzePage
              apiBaseUrl={apiBaseUrl}
              health={health}
              onAnalysisComplete={complete}
            />
          )}
          {tab === 'results' && result && file && preview && (
            <ResultsPage
              result={result}
              originalFile={file}
              previewUrl={preview}
              onReset={() => {
                setResult(null);
                setFile(null);
                setPreview(null);
                navigate('analyze');
              }}
            />
          )}
          {tab === 'history' && <HistoryPage />}
          {tab === 'methodology' && <MethodologyPage />}
          {tab === 'models' && <ModelsPage apiBaseUrl={apiBaseUrl} />}
          {tab === 'settings' && (
            <SettingsPage
              apiBaseUrl={apiBaseUrl}
              setApiBaseUrl={setApiBaseUrl}
              health={health}
              checking={checking}
              onCheck={refreshHealth}
            />
          )}
        </main>
        <DisclaimerBanner />
        <Footer onViewModels={() => navigate('models')} />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <HistoryProvider>
          <MainApp />
        </HistoryProvider>
      </ErrorBoundary>
    </ThemeProvider>
  );
}
