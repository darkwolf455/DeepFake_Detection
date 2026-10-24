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

// Error Boundary component
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

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col items-center justify-center p-6 text-center">
          <div className="max-w-md p-8 rounded-3xl bg-slate-900 border border-rose-500/30 space-y-4">
            <h2 className="text-xl font-bold text-rose-400">Something went wrong</h2>
            <p className="text-xs text-slate-400 font-mono">
              {this.state.error?.message || 'An unexpected rendering error occurred.'}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-cyan-500 text-slate-950 font-bold hover:bg-cyan-400 transition-colors"
            >
              Reload Application
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

  // Settings
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

    // Save record to local history
    addHistoryItem({
      id: result.id,
      filename: file.name,
      modality: result.modality,
      verdict: result.verdict,
      confidence: result.confidence,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
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

  return (
    <div className="min-h-screen flex flex-col bg-[#0B0F19] text-slate-100 font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Persistent Forensics Notice Banner */}
      <DisclaimerBanner />

      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasActiveResult={!!activeResult}
      />

      {/* Main Page Content */}
      <main className="flex-1 w-full">
        {activeTab === 'analyze' && (
          <AnalyzePage
            onAnalysisComplete={handleAnalysisComplete}
            apiBaseUrl={apiBaseUrl}
          />
        )}

        {activeTab === 'results' && activeResult && activeFile && activePreviewUrl && (
          <ResultsPage
            result={activeResult}
            originalFile={activeFile}
            previewUrl={activePreviewUrl}
            onReset={handleReset}
          />
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

      {/* Footer */}
      <Footer onNavigateToCode={() => setActiveTab('code')} />
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
