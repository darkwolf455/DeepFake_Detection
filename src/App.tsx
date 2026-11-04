
import React, { useState } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { HistoryProvider, useHistory } from './context/HistoryContext';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { AnalyzePage } from './pages/AnalyzePage';
import { ResultsPage } from './pages/ResultsPage';
import { HistoryPage } from './pages/HistoryPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { CodeExplorerPage } from './pages/CodeExplorerPage';
import { SettingsPage } from './pages/SettingsPage';
import { AnalyzeResponse } from './types';

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-[var(--bg)] p-6 text-[var(--text)]">
          <div className="w-full max-w-md rounded-2xl border border-[var(--danger)]/30 bg-[var(--surface)] p-8 shadow-xl">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--danger-soft)] text-[var(--danger)]">
              <span className="text-2xl font-bold">!</span>
            </div>

            <h2 className="text-xl font-bold">Something went wrong</h2>

            <p className="mt-2 break-words font-mono text-xs leading-6 text-[var(--text-secondary)]">
              {this.state.error?.message ||
                'An unexpected rendering error occurred.'}
            </p>

            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="mt-6 rounded-xl bg-[var(--accent)] px-5 py-3 text-sm font-bold text-white transition hover:opacity-90"
            >
              Reload application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function MainApp() {
  const [activeTab, setActiveTab] = useState<string>('analyze');
  const [activeResult, setActiveResult] = useState<AnalyzeResponse | null>(null);
  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [activePreviewUrl, setActivePreviewUrl] = useState<string | null>(null);

  const [apiBaseUrl, setApiBaseUrl] = useState<string>('');
  const [inconclusiveLower, setInconclusiveLower] = useState<number>(0.40);
  const [inconclusiveUpper, setInconclusiveUpper] = useState<number>(0.65);

  const { addHistoryItem } = useHistory();

  const handleAnalysisComplete = (
    result: AnalyzeResponse,
    file: File,
    previewUrl: string
  ) => {
    setActiveResult(result);
    setActiveFile(file);
    setActivePreviewUrl(previewUrl);

    addHistoryItem({
      id: result.id,
      filename: file.name,
      modality: result.modality,
      verdict: result.verdict,
      confidence: result.confidence,
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
      scores: result.scores,
      file_sha256: result.file_sha256,
    });

    setActiveTab('results');
  };

  const handleReset = () => {
    setActiveResult(null);
    setActiveFile(null);

    if (activePreviewUrl) {
      URL.revokeObjectURL(activePreviewUrl);
      setActivePreviewUrl(null);
    }

    setActiveTab('analyze');
  };

  const pageTitles: Record<string, string> = {
    analyze: 'Analyze media',
    results: 'Analysis results',
    history: 'Investigation history',
    methodology: 'Detection methodology',
    code: 'Code explorer',
    settings: 'System settings',
  };

  return (
    <div className="flex min-h-screen bg-[var(--bg)] font-sans text-[var(--text)] selection:bg-[var(--accent)]/20">
      {/* Persistent desktop sidebar + mobile navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasActiveResult={!!activeResult}
      />

      {/* Main application area */}
      <div className="flex min-h-screen min-w-0 flex-1 flex-col lg:ml-[264px]">
        <DisclaimerBanner />

        {/* Workspace topbar */}
        <header className="sticky top-0 z-30 border-b border-[var(--border)] bg-[var(--surface)]/90 backdrop-blur-xl">
          <div className="flex min-h-[62px] items-center justify-between gap-4 px-4 sm:px-7 lg:px-9">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.15em] text-[var(--muted)]">
                <span>VeriSense</span>
                <span className="text-[var(--border-strong)]">/</span>
                <span className="truncate text-[var(--accent)]">
                  {pageTitles[activeTab] || 'Workspace'}
                </span>
              </div>

              <p className="mt-1 truncate text-xs font-semibold text-[var(--text-secondary)]">
                Multimodal media forensics workspace
              </p>
            </div>

            <div className="hidden items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 sm:flex">
              <span className="h-2 w-2 rounded-full bg-[var(--success)]" />
              <span className="text-[10px] font-bold tracking-wide text-[var(--text-secondary)]">
                FORENSIC WORKSPACE
              </span>
            </div>
          </div>
        </header>

        <main className="min-w-0 flex-1">
          {activeTab === 'analyze' && (
            <AnalyzePage
              onAnalysisComplete={handleAnalysisComplete}
              apiBaseUrl={apiBaseUrl}
            />
          )}

          {activeTab === 'results' &&
            activeResult &&
            activeFile &&
            activePreviewUrl && (
              <ResultsPage
                result={activeResult}
                originalFile={activeFile}
                previewUrl={activePreviewUrl}
                onReset={handleReset}
              />
            )}

          {activeTab === 'results' && !activeResult && (
            <div className="mx-auto max-w-3xl px-6 py-20 text-center">
              <h2 className="text-xl font-bold">No analysis available</h2>
              <p className="mt-2 text-sm text-[var(--text-secondary)]">
                Upload a media file and run an analysis to view its results.
              </p>
              <button
                onClick={() => setActiveTab('analyze')}
                className="mt-5 rounded-xl bg-[var(--accent)] px-5 py-3 text-sm font-bold text-white"
              >
                Start an analysis
              </button>
            </div>
          )}

          {activeTab === 'history' && <HistoryPage />}

          {activeTab === 'methodology' && (
            <MethodologyPage onExploreCode={() => setActiveTab('code')} />
          )}

          {activeTab === 'code' && <CodeExplorerPage />}

          {activeTab === 'settings' && (
            <SettingsPage
              apiBaseUrl={apiBaseUrl}
              setApiBaseUrl={setApiBaseUrl}
              inconclusiveLower={inconclusiveLower}
              setInconclusiveLower={setInconclusiveLower}
              inconclusiveUpper={inconclusiveUpper}
              setInconclusiveUpper={setInconclusiveUpper}
            />
          )}
        </main>

        <Footer onNavigateToCode={() => setActiveTab('code')} />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <HistoryProvider>
          <MainApp />
        </HistoryProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
