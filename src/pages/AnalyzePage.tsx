import React, { useState, useRef } from 'react';
import {
  Upload,
  FileImage,
  FileVideo,
  FileAudio,
  Sparkles,
  AlertCircle,
  X,
  ArrowRight,
  CheckCircle2,
  Loader2,
  StopCircle,
  RefreshCw,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { analyzeMediaFile } from '../services/api';
import { AnalyzeResponse, Modality } from '../types';
import { AudioWaveform } from '../components/AudioWaveform';

interface AnalyzePageProps {
  onAnalysisComplete: (result: AnalyzeResponse, file: File, previewUrl: string) => void;
  apiBaseUrl: string;
}

const PIPELINE_STEPS = [
  'Extracting media frames & 16kHz audio waveforms',
  'Computing CLIP ViT-L/14 & Wav2Vec2 representations',
  'Analyzing temporal artifacts & SyncNet lip synchronization',
  'Executing Platt-calibrated late fusion & Grad-CAM map',
];

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

  // Client-side MIME & extension detection
  const inspectFile = (file: File) => {
    setErrorMessage(null);
    const ext = file.name.split('.').pop()?.toLowerCase() || '';

    let mod: Modality | null = null;
    if (file.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext)) {
      mod = 'image';
    } else if (file.type.startsWith('video/') || ['mp4', 'mov', 'avi', 'webm', 'mkv'].includes(ext)) {
      mod = 'video';
    } else if (file.type.startsWith('audio/') || ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'].includes(ext)) {
      mod = 'audio';
    } else {
      setErrorMessage(`Unsupported media format for '${file.name}'. Please upload an image, video, or audio file.`);
      return;
    }

    if (file.size > 150 * 1024 * 1024) {
      setErrorMessage('File size exceeds 150MB limit.');
      return;
    }

    setSelectedFile(file);
    setDetectedModality(mod);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      inspectFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setDetectedModality(null);
    setErrorMessage(null);
    setIsAnalyzing(false);
    setCurrentStepIndex(0);
  };

  const startAnalysis = async () => {
    if (!selectedFile || !previewUrl) return;

    setIsAnalyzing(true);
    setErrorMessage(null);
    setCurrentStepIndex(0);

    // Create abort controller
    abortControllerRef.current = new AbortController();

    // Step animation timer
    const stepInterval = setInterval(() => {
      setCurrentStepIndex((prev) => {
        if (prev < PIPELINE_STEPS.length - 1) return prev + 1;
        return prev;
      });
    }, 600);

    try {
      const response = await analyzeMediaFile(
        selectedFile,
        apiBaseUrl,
        abortControllerRef.current.signal
      );
      clearInterval(stepInterval);
      setCurrentStepIndex(PIPELINE_STEPS.length - 1);
      setTimeout(() => {
        setIsAnalyzing(false);
        onAnalysisComplete(response, selectedFile, previewUrl);
      }, 400);
    } catch (err: any) {
      clearInterval(stepInterval);
      setIsAnalyzing(false);
      if (err.errorCode !== 'ERR_ABORTED') {
        setErrorMessage(err.message || 'Analysis failed. Check your network or server status.');
      }
    }
  };

  // Sample Preset Media generators for rapid demonstration
  const loadPreset = async (presetType: 'image_fake' | 'image_real' | 'video_fake' | 'audio_fake') => {
    handleClear();

    if (presetType === 'image_fake') {
      // Create SVG blob simulating AI generator face artifacts
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
        <defs>
          <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#1e1b4b" />
            <stop offset="100%" stop-color="#31104b" />
          </linearGradient>
          <radialGradient id="face" cx="50%" cy="45%" r="35%">
            <stop offset="0%" stop-color="#ffd5b5" />
            <stop offset="70%" stop-color="#e09f67" />
            <stop offset="100%" stop-color="#b0632e" />
          </radialGradient>
        </defs>
        <rect width="400" height="400" fill="url(#bg)" />
        <ellipse cx="200" cy="180" rx="90" ry="110" fill="url(#face)" />
        <circle cx="165" cy="160" r="14" fill="#1e293b" />
        <circle cx="235" cy="160" r="14" fill="#1e293b" />
        <circle cx="168" cy="158" r="5" fill="#38bdf8" />
        <circle cx="238" cy="158" r="5" fill="#38bdf8" />
        <ellipse cx="200" cy="235" rx="30" ry="12" fill="#991b1b" />
        <text x="200" y="340" fill="#a855f7" font-family="monospace" font-size="14" text-anchor="middle">SYNTHESIZED_FACE_DIFFUSION_SAMPLE</text>
      </svg>`;
      const blob = new Blob([svg], { type: 'image/svg+xml' });
      const file = new File([blob], 'deepfake_ai_generated_portrait.svg', { type: 'image/svg+xml' });
      inspectFile(file);
    } else if (presetType === 'image_real') {
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
        <rect width="400" height="400" fill="#0f172a" />
        <circle cx="200" cy="180" r="85" fill="#fbcfe8" />
        <circle cx="170" cy="165" r="10" fill="#0284c7" />
        <circle cx="230" cy="165" r="10" fill="#0284c7" />
        <path d="M 180 230 Q 200 250 220 230" stroke="#be185d" stroke-width="4" fill="none" />
        <text x="200" y="340" fill="#10b981" font-family="monospace" font-size="14" text-anchor="middle">AUTHENTIC_CAMERA_RAW_SAMPLE</text>
      </svg>`;
      const blob = new Blob([svg], { type: 'image/svg+xml' });
      const file = new File([blob], 'authentic_camera_photo.svg', { type: 'image/svg+xml' });
      inspectFile(file);
    } else if (presetType === 'video_fake') {
      // Empty minimal webm or mp4 blob
      const header = new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 0x9f, 0x42, 0x86, 0x81, 0x01, 0x42, 0xf7, 0x81, 0x01]);
      const blob = new Blob([header], { type: 'video/webm' });
      const file = new File([blob], 'faceforensics_deepfake_sequence.webm', { type: 'video/webm' });
      inspectFile(file);
    } else if (presetType === 'audio_fake') {
      // Create minimal valid WAV header with tone
      const sampleRate = 16000;
      const numSamples = sampleRate * 4; // 4 seconds
      const buffer = new ArrayBuffer(44 + numSamples * 2);
      const view = new DataView(buffer);
      // RIFF identifier
      view.setUint32(0, 0x52494646, false); // "RIFF"
      view.setUint32(4, 36 + numSamples * 2, true);
      view.setUint32(8, 0x57415645, false); // "WAVE"
      view.setUint32(12, 0x666d7420, false); // "fmt "
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM
      view.setUint16(22, 1, true); // Mono
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true); // 16-bit
      view.setUint32(36, 0x64617461, false); // "data"
      view.setUint32(40, numSamples * 2, true);

      for (let i = 0; i < numSamples; i++) {
        const sample = Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 0.4 * 32767;
        view.setInt16(44 + i * 2, sample, true);
      }

      const blob = new Blob([buffer], { type: 'audio/wav' });
      const file = new File([blob], 'asvspoof_tts_cloned_voice.wav', { type: 'audio/wav' });
      inspectFile(file);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12">
      {/* Hero Header */}
      <div className="text-center max-w-2xl mx-auto mb-8 sm:mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold mb-4">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Multimodal Deepfake Forensic Engine</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-3">
          Authenticate Media with{' '}
          <span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-violet-400 bg-clip-text text-transparent">
            VeriSense
          </span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
          Upload an image, video, or audio clip to detect AI generation, face-swaps, synthetic speech cloning, and desynchronized lip-sync anomalies.
        </p>
      </div>

      {/* Preset Quick-Test Bar */}
      <div className="flex flex-wrap items-center justify-center gap-2 mb-8">
        <span className="text-xs text-slate-500 font-mono mr-1">Quick Test Presets:</span>
        <button
          onClick={() => loadPreset('image_fake')}
          className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-xs transition-colors flex items-center gap-1.5"
        >
          <FileImage className="w-3.5 h-3.5 text-rose-400" />
          <span>Synthetic AI Face</span>
        </button>
        <button
          onClick={() => loadPreset('image_real')}
          className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-xs transition-colors flex items-center gap-1.5"
        >
          <FileImage className="w-3.5 h-3.5 text-emerald-400" />
          <span>Authentic Camera Photo</span>
        </button>
        <button
          onClick={() => loadPreset('video_fake')}
          className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-xs transition-colors flex items-center gap-1.5"
        >
          <FileVideo className="w-3.5 h-3.5 text-violet-400" />
          <span>FaceForensics Video Clip</span>
        </button>
        <button
          onClick={() => loadPreset('audio_fake')}
          className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-xs transition-colors flex items-center gap-1.5"
        >
          <FileAudio className="w-3.5 h-3.5 text-amber-400" />
          <span>ASVspoof Voice Clone</span>
        </button>
      </div>

      {/* Error Alert Banner */}
      <AnimatePresence>
        {errorMessage && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-6 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center justify-between gap-3 shadow-lg shadow-rose-900/10"
          >
            <div className="flex items-center gap-2.5">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="p-1 rounded-lg hover:bg-rose-500/20 text-rose-400"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Upload / Preview Container */}
      {!selectedFile ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative rounded-3xl border-2 border-dashed transition-all duration-300 p-10 sm:p-16 flex flex-col items-center justify-center text-center cursor-pointer group ${
            isDragging
              ? 'border-cyan-400 bg-cyan-950/20 shadow-2xl shadow-cyan-500/20'
              : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="image/*,video/*,audio/*,.mp4,.mov,.webm,.wav,.mp3,.flac,.jpg,.jpeg,.png,.webp"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                inspectFile(e.target.files[0]);
              }
            }}
          />

          <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-slate-800 to-slate-900 border border-slate-700 flex items-center justify-center mb-6 group-hover:scale-110 group-hover:border-cyan-500/40 transition-all duration-300 shadow-xl">
            <Upload className="w-9 h-9 text-cyan-400 group-hover:text-cyan-300" />
          </div>

          <h3 className="text-xl font-bold text-white mb-2">
            Drag & drop media file or{' '}
            <span className="text-cyan-400 underline decoration-cyan-500/40 underline-offset-4">
              browse local files
            </span>
          </h3>

          <p className="text-slate-400 text-xs sm:text-sm max-w-md mb-6">
            Supports Images (JPEG, PNG, WebP), Videos (MP4, WebM, MOV), and Audio (WAV, MP3, FLAC) up to 150MB.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-mono text-slate-500">
            <span className="flex items-center gap-1.5">
              <FileImage className="w-4 h-4 text-cyan-400" /> CLIP ViT-L/14
            </span>
            <span className="flex items-center gap-1.5">
              <FileVideo className="w-4 h-4 text-violet-400" /> Temporal Transformer
            </span>
            <span className="flex items-center gap-1.5">
              <FileAudio className="w-4 h-4 text-amber-400" /> Wav2Vec2 XLS-R
            </span>
          </div>
        </div>
      ) : (
        /* Preview Card */
        <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-800">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                {detectedModality === 'image' && <FileImage className="w-5 h-5" />}
                {detectedModality === 'video' && <FileVideo className="w-5 h-5" />}
                {detectedModality === 'audio' && <FileAudio className="w-5 h-5" />}
              </div>
              <div>
                <h4 className="text-sm font-bold text-white truncate max-w-xs sm:max-w-md">
                  {selectedFile.name}
                </h4>
                <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                  <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                  <span>•</span>
                  <span className="uppercase text-cyan-400 font-semibold">{detectedModality}</span>
                </div>
              </div>
            </div>

            {!isAnalyzing && (
              <button
                onClick={handleClear}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
              >
                <X className="w-4 h-4" />
                <span>Change File</span>
              </button>
            )}
          </div>

          {/* Media Player / Viewer Preview */}
          <div className="mb-6 rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[240px] max-h-[460px]">
            {detectedModality === 'image' && (
              <img
                src={previewUrl!}
                alt="Selected preview"
                className="max-h-[460px] w-auto object-contain"
              />
            )}
            {detectedModality === 'video' && (
              <video
                src={previewUrl!}
                controls
                className="max-h-[460px] w-auto object-contain"
              />
            )}
            {detectedModality === 'audio' && (
              <div className="w-full p-6">
                <AudioWaveform audioUrl={previewUrl!} />
              </div>
            )}
          </div>

          {/* Stepper Pipeline Progress Indicator (during analysis) */}
          {isAnalyzing && (
            <div className="mb-6 p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs font-semibold text-cyan-400 uppercase tracking-wider font-mono">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Pipeline Execution Status</span>
                </div>
                <button
                  onClick={handleClear}
                  className="flex items-center gap-1 text-xs text-rose-400 hover:text-rose-300"
                >
                  <StopCircle className="w-3.5 h-3.5" />
                  <span>Cancel</span>
                </button>
              </div>

              <div className="space-y-2 pt-2">
                {PIPELINE_STEPS.map((step, idx) => {
                  const isDone = idx < currentStepIndex;
                  const isCurrent = idx === currentStepIndex;
                  return (
                    <div
                      key={idx}
                      className={`flex items-center gap-3 text-xs transition-colors duration-200 ${
                        isDone
                          ? 'text-emerald-400 font-medium'
                          : isCurrent
                          ? 'text-cyan-300 font-semibold'
                          : 'text-slate-600'
                      }`}
                    >
                      {isDone ? (
                        <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
                      ) : isCurrent ? (
                        <Loader2 className="w-4 h-4 shrink-0 animate-spin text-cyan-400" />
                      ) : (
                        <div className="w-4 h-4 rounded-full border border-slate-700 shrink-0" />
                      )}
                      <span>{step}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Launch Action Button */}
          {!isAnalyzing ? (
            <button
              onClick={startAnalysis}
              className="w-full py-4 rounded-2xl font-bold text-sm tracking-wide bg-gradient-to-r from-cyan-500 via-sky-500 to-violet-600 hover:from-cyan-400 hover:to-violet-500 text-white shadow-xl shadow-cyan-500/25 flex items-center justify-center gap-2 active:scale-[0.99] transition-all"
            >
              <span>Initiate Deepfake Forensic Scan</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              disabled
              className="w-full py-4 rounded-2xl font-bold text-sm tracking-wide bg-slate-800 text-slate-400 flex items-center justify-center gap-2 cursor-not-allowed"
            >
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span>Analyzing Multi-Modal Neural Features...</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
};
