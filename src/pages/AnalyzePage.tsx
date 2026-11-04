import React, { useRef, useState } from 'react';
import {
  Upload,
  FileImage,
  FileVideo,
  FileAudio,
  AlertCircle,
  X,
  Loader2,
  StopCircle,
  RefreshCw,
  CheckCircle2,
  Sparkles,
  Cpu,
  ShieldCheck,
  Zap,
  LockKeyhole,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { analyzeMediaFile } from '../services/api';
import { AnalyzeResponse, Modality } from '../types';
import { AudioWaveform } from '../components/AudioWaveform';

interface AnalyzePageProps {
  onAnalysisComplete: (
    result: AnalyzeResponse,
    file: File,
    previewUrl: string
  ) => void;
  apiBaseUrl: string;
}

const PIPELINE_STEPS = [
  'Extracting media frames & 16kHz audio waveforms',
  'Computing CLIP ViT-L/14 & Wav2Vec2 representations',
  'Analyzing temporal artifacts & SyncNet lip synchronization',
  'Executing Platt-calibrated late fusion & Grad-CAM map',
];

const MODULE_HIGHLIGHTS = [
  { title: 'Visual Forensics', desc: 'CLIP ViT-L/14 spatial artifact detection', icon: Cpu, color: '#f47721' },
  { title: 'Temporal Analysis', desc: 'Frame consistency & manipulation checks', icon: Sparkles, color: '#9b7be7' },
  { title: 'Audio Intelligence', desc: 'Wav2Vec2 XLS-R synthetic voice cloning', icon: FileAudio, color: '#d99a42' },
];

const formatFileSize = (bytes: number) => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

export const AnalyzePage: React.FC<AnalyzePageProps> = ({
  onAnalysisComplete,
  apiBaseUrl,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectedModality, setDetectedModality] = useState<Modality | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const inspectFile = (file: File) => {
    setErrorMessage(null);

    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    let modality: Modality | null = null;

    if (
      file.type.startsWith('image/') ||
      ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext)
    ) {
      modality = 'image';
    } else if (
      file.type.startsWith('video/') ||
      ['mp4', 'mov', 'avi', 'webm', 'mkv'].includes(ext)
    ) {
      modality = 'video';
    } else if (
      file.type.startsWith('audio/') ||
      ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'].includes(ext)
    ) {
      modality = 'audio';
    }

    if (!modality) {
      setErrorMessage(
        `Unsupported media format for '${file.name}'. Please upload an image, video, or audio file.`
      );
      return;
    }

    if (file.size > 150 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 150MB limit.');
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    const url = URL.createObjectURL(file);

    setSelectedFile(file);
    setDetectedModality(modality);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files.length > 0) {
      inspectFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setSelectedFile(null);
    setPreviewUrl(null);
    setDetectedModality(null);
    setErrorMessage(null);
    setIsAnalyzing(false);
    setCurrentStepIndex(0);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const startAnalysis = async () => {
    if (!selectedFile || !previewUrl || isAnalyzing) return;

    setIsAnalyzing(true);
    setErrorMessage(null);
    setCurrentStepIndex(0);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const stepInterval = window.setInterval(() => {
      setCurrentStepIndex((prev) =>
        prev < PIPELINE_STEPS.length - 1 ? prev + 1 : prev
      );
    }, 600);

    try {
      const response = await analyzeMediaFile(
        selectedFile,
        apiBaseUrl,
        controller.signal
      );

      window.clearInterval(stepInterval);
      setCurrentStepIndex(PIPELINE_STEPS.length - 1);
      setIsAnalyzing(false);

      onAnalysisComplete(response, selectedFile, previewUrl);
    } catch (err: any) {
      window.clearInterval(stepInterval);
      setIsAnalyzing(false);

      if (err?.errorCode !== 'ERR_ABORTED') {
        setErrorMessage(
          err?.message ||
            'Analysis failed. Check your network or server status.'
        );
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  const ModalityIcon =
    detectedModality === 'image'
      ? FileImage
      : detectedModality === 'video'
        ? FileVideo
        : FileAudio;

  return (
    <main className="dashboard-bg min-h-screen">
      {/* Expanded container to utilize full screen width professionally */}
      <div className="mx-auto max-w-[1520px] px-6 py-8 sm:px-8 lg:px-12 lg:py-10">
        
        {/* Top Header Bar */}
        <div className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
          <div>
            <div className="mb-2 flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest text-[var(--muted)]">
              <span>Verisense</span>
              <span>/</span>
              <span className="text-[var(--accent)]">Forensic Workspace</span>
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight text-[var(--text)] sm:text-4xl">
              DeepFake Media Analysis
            </h1>
            <p className="mt-1 text-sm text-[var(--text-secondary)]">
              Examine images, videos, or audio files for synthetic manipulation and identity anomalies.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-xs font-semibold text-[var(--text-secondary)] shadow-sm">
              <LockKeyhole className="h-4 w-4 text-[var(--success)]" />
              Secure Session Active
            </div>
          </div>
        </div>

        {/* Error Alert */}
        <AnimatePresence>
          {errorMessage && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="mb-6 flex items-center justify-between gap-3 rounded-xl border border-[var(--danger)]/30 bg-[var(--danger-soft)] p-4 text-sm text-[var(--danger)] shadow-sm"
            >
              <div className="flex items-center gap-3">
                <AlertCircle className="h-5 w-5 shrink-0" />
                <span>{errorMessage}</span>
              </div>
              <button
                onClick={() => setErrorMessage(null)}
                className="rounded-lg p-1 hover:bg-[var(--danger)]/10"
                aria-label="Dismiss error"
              >
                <X className="h-4 w-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Grid Layout to fill screen evenly */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-12 items-start">
          
          {/* Main Upload / Preview Area (Takes 8 columns) */}
          <div className="lg:col-span-8">
            {!selectedFile ? (
              <div className="vs-card overflow-hidden p-8 sm:p-12 shadow-md">
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept="image/*,video/*,audio/*,.mp4,.mov,.webm,.wav,.mp3,.flac,.jpg,.jpeg,.png,.webp"
                  onChange={(e) => {
                    if (e.target.files?.[0]) {
                      inspectFile(e.target.files[0]);
                    }
                    e.target.value = '';
                  }}
                />

                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={(e) => {
                    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
                      setIsDragging(false);
                    }
                  }}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      fileInputRef.current?.click();
                    }
                  }}
                  className={`group flex min-h-[420px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 transition-all duration-300 ${
                    isDragging
                      ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                      : 'border-[var(--border-strong)] bg-[var(--surface-2)] hover:border-[var(--accent)] hover:bg-[var(--surface-hover)]'
                  }`}
                >
                  <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)] shadow-sm transition-transform group-hover:scale-105">
                    <Upload className="h-9 w-9" />
                  </div>

                  <h3 className="text-xl font-bold tracking-tight text-[var(--text)]">
                    {isDragging ? 'Drop media file here' : 'Drag & Drop media here'}
                  </h3>
                  <p className="mt-2 text-sm text-[var(--text-secondary)]">
                    or click to browse from your device <span className="font-mono text-[var(--muted)]">(Max size: 150MB)</span>
                  </p>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="vs-button-primary mt-8 px-6 py-3 text-sm font-bold shadow-md"
                  >
                    Select Media File
                  </button>

                  <div className="mt-8 flex flex-wrap justify-center gap-3">
                    {['JPG / PNG / WEBP', 'MP4 / MOV / WEBM', 'WAV / MP3 / FLAC'].map((type) => (
                      <span
                        key={type}
                        className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3.5 py-1.5 font-mono text-[11px] font-semibold text-[var(--muted)]"
                      >
                        {type}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="vs-card space-y-6 p-6 sm:p-8 shadow-md">
                <div className="flex items-center justify-between border-b border-[var(--border)] pb-5">
                  <div className="flex items-center gap-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--accent-soft)] text-[var(--accent)] shadow-sm">
                      <ModalityIcon className="h-6 w-6" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-[var(--text)]">
                        {selectedFile.name}
                      </h3>
                      <div className="mt-1 flex items-center gap-2 text-xs font-mono text-[var(--muted)]">
                        <span>{formatFileSize(selectedFile.size)}</span>
                        <span>•</span>
                        <span className="font-bold text-[var(--accent)] uppercase">{detectedModality}</span>
                        <span>•</span>
                        <span className="text-[var(--success)] flex items-center gap-1 font-semibold">
                          <CheckCircle2 className="h-3.5 w-3.5" /> Ready for examination
                        </span>
                      </div>
                    </div>
                  </div>

                  {!isAnalyzing && (
                    <button
                      onClick={handleClear}
                      className="vs-button-secondary px-4 py-2 text-xs"
                    >
                      <RefreshCw className="mr-2 h-3.5 w-3.5" />
                      Change File
                    </button>
                  )}
                </div>

                {/* Media Preview Box */}
                <div className="flex min-h-[340px] max-h-[520px] items-center justify-center overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4 shadow-inner">
                  {detectedModality === 'image' && previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Selected media preview"
                      className="max-h-[460px] w-auto max-w-full rounded-lg object-contain shadow-sm"
                    />
                  )}

                  {detectedModality === 'video' && previewUrl && (
                    <video
                      src={previewUrl}
                      controls
                      className="max-h-[460px] w-auto max-w-full rounded-lg shadow-sm"
                    />
                  )}

                  {detectedModality === 'audio' && previewUrl && (
                    <div className="w-full max-w-md p-6 bg-[var(--surface)] rounded-xl border border-[var(--border)] shadow-sm">
                      <AudioWaveform audioUrl={previewUrl} />
                    </div>
                  )}
                </div>

                {/* Analysis Progress or Start Button */}
                <div className="pt-2">
                  {isAnalyzing ? (
                    <div className="space-y-4 rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-6 shadow-sm">
                      <div className="flex items-center justify-between text-xs font-semibold">
                        <span className="flex items-center gap-2.5 text-[var(--accent)]">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Executing Multimodal Forensic Pipeline...
                        </span>
                        <button
                          onClick={handleClear}
                          className="flex items-center gap-1 text-[var(--danger)] hover:underline"
                        >
                          <StopCircle className="h-4 w-4" /> Cancel Analysis
                        </button>
                      </div>

                      <div className="rounded-lg bg-[var(--surface)] p-4 border border-[var(--border)]">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-bold text-[var(--text)] font-mono">
                            STEP {currentStepIndex + 1} OF {PIPELINE_STEPS.length}
                          </span>
                          <Cpu className="h-4 w-4 text-[var(--accent)] animate-pulse" />
                        </div>
                        <p className="text-xs text-[var(--text-secondary)] font-mono">
                          {PIPELINE_STEPS[currentStepIndex]}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={startAnalysis}
                      className="vs-button-primary w-full justify-center py-4 text-base font-bold shadow-lg"
                    >
                      <Sparkles className="mr-2.5 h-5 w-5" />
                      Start Forensic Analysis
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Sidebar Info / Intelligence Modules (Takes 4 columns to balance layout) */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* Engine Status Card */}
            <div className="vs-card p-6 shadow-md">
              <div className="flex items-center justify-between border-b border-[var(--border)] pb-4">
                <h3 className="text-sm font-bold text-[var(--text)]">Engine Status</h3>
                <span className="flex items-center gap-2 text-xs font-bold text-[var(--success)]">
                  <span className="h-2.5 w-2.5 rounded-full bg-[var(--success)] animate-pulse" />
                  Online
                </span>
              </div>
              <div className="mt-4 space-y-3 text-xs text-[var(--text-secondary)]">
                <div className="flex justify-between">
                  <span>Active Models</span>
                  <span className="font-mono font-bold text-[var(--text)]">CLIP, Wav2Vec2, SyncNet</span>
                </div>
                <div className="flex justify-between">
                  <span>Max File Size</span>
                  <span className="font-mono font-bold text-[var(--text)]">150 MB</span>
                </div>
                <div className="flex justify-between">
                  <span>Inference Latency</span>
                  <span className="font-mono font-bold text-[var(--success)]">~1.2s avg</span>
                </div>
              </div>
            </div>

            {/* Active Intelligence Modules */}
            <div className="vs-card p-6 shadow-md">
              <div className="mb-4 flex items-center justify-between border-b border-[var(--border)] pb-4">
                <h3 className="text-sm font-bold text-[var(--text)]">Active Intelligence Modules</h3>
                <Zap className="h-4 w-4 text-[var(--accent)]" />
              </div>
              <div className="space-y-4">
                {MODULE_HIGHLIGHTS.map((mod) => {
                  const Icon = mod.icon;
                  return (
                    <div key={mod.title} className="flex items-start gap-3 rounded-xl bg-[var(--surface-2)] p-3 border border-[var(--border)]">
                      <div
                        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
                        style={{ color: mod.color, backgroundColor: `${mod.color}15` }}
                      >
                        <Icon className="h-4 w-4" />
                      </div>
                      <div>
                        <h4 className="text-xs font-bold text-[var(--text)]">{mod.title}</h4>
                        <p className="mt-0.5 text-[11px] text-[var(--text-secondary)]">{mod.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>

        </div>

      </div>
    </main>
  );
};