import React, { useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  AlertOctagon,
  HelpCircle,
  Layers,
  FileText,
  RotateCcw,
  Cpu,
  Hash,
} from 'lucide-react';
import { motion } from 'motion/react';
import { AnalyzeResponse } from '../types';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { GradCamViewer } from '../components/GradCamViewer';
import { AudioWaveform } from '../components/AudioWaveform';
import { ReportModal } from '../components/ReportModal';

interface ResultsPageProps {
  result: AnalyzeResponse;
  originalFile: File;
  previewUrl: string;
  onReset: () => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({
  result,
  originalFile,
  previewUrl,
  onReset,
}) => {
  const [showReportModal, setShowReportModal] = useState(false);
  const [seekTime, setSeekTime] = useState<number | null>(null);

  const verdictConfig = {
    authentic: {
      badgeBg: 'bg-[var(--success-soft)] text-[var(--success)] border-transparent',
      icon: ShieldCheck,
      headline: 'Authentic Media Verified',
      summary: 'No significant synthetic diffusion artifacts or spatial anomalies detected. Media aligns with verified natural signatures.',
    },
    manipulated: {
      badgeBg: 'bg-[var(--danger-soft)] text-[var(--danger)] border-transparent',
      icon: AlertOctagon,
      headline: 'Manipulated / Deepfake Content Detected',
      summary: 'Substantial synthetic artifacts identified across spatial attention tokens and frequency spectrums consistent with generative manipulation.',
    },
    inconclusive: {
      badgeBg: 'bg-[var(--warning-soft)] text-[var(--warning)] border-transparent',
      icon: HelpCircle,
      headline: 'Inconclusive Forensic Verdict',
      summary: 'Certainty score falls within the indeterminate margin. Compression or resolution limits prevent a definitive determination.',
    },
  }[result.verdict];

  const VerdictIcon = verdictConfig.icon;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 sm:py-12 space-y-8">
      
      {/* Top Header & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[var(--border)]">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono font-semibold uppercase tracking-wider text-[var(--accent)]">
            <span>Report #{result.id}</span>
            <span>•</span>
            <span className="text-[var(--text-secondary)]">{result.modality}</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-[var(--text)] mt-1 tracking-tight">
            Forensic Inspection Dashboard
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowReportModal(true)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-[var(--accent-soft)] text-[var(--accent)] border border-[var(--border-strong)] hover:opacity-90 transition-all shadow-sm"
          >
            <FileText className="w-4 h-4" />
            <span>Forensic Certificate</span>
          </button>
          <button
            onClick={onReset}
            className="vs-button-secondary px-4 py-2.5 text-xs font-bold"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Analyze Another</span>
          </button>
        </div>
      </div>

      {/* Main Verdict & Gauge Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* Left Verdict Box (8 cols) */}
        <div className="vs-card lg:col-span-8 p-6 sm:p-8 flex flex-col justify-between shadow-md">
          <div>
            <div className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${verdictConfig.badgeBg} mb-5`}>
              <VerdictIcon className="w-4 h-4" />
              <span>Verdict: {result.verdict}</span>
            </div>

            <h3 className="text-2xl sm:text-3xl font-black text-[var(--text)] mb-3 tracking-tight">
              {verdictConfig.headline}
            </h3>

            <p className="text-[var(--text-secondary)] text-sm sm:text-base leading-relaxed mb-8">
              {verdictConfig.summary}
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-6 border-t border-[var(--border)] font-mono text-xs">
            <div className="rounded-xl bg-[var(--surface-2)] p-3 border border-[var(--border)]">
              <span className="text-[var(--muted)] block text-[10px] uppercase font-bold">Latency</span>
              <span className="text-[var(--text)] font-extrabold mt-0.5 block">{result.processing_ms} ms</span>
            </div>
            <div className="rounded-xl bg-[var(--surface-2)] p-3 border border-[var(--border)]">
              <span className="text-[var(--muted)] block text-[10px] uppercase font-bold">File Name</span>
              <span className="text-[var(--text)] font-extrabold truncate block mt-0.5">{originalFile.name}</span>
            </div>
            <div className="rounded-xl bg-[var(--surface-2)] p-3 border border-[var(--border)]">
              <span className="text-[var(--muted)] block text-[10px] uppercase font-bold">File Size</span>
              <span className="text-[var(--text)] font-extrabold mt-0.5 block">{(originalFile.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
            <div className="rounded-xl bg-[var(--surface-2)] p-3 border border-[var(--border)]">
              <span className="text-[var(--muted)] block text-[10px] uppercase font-bold">Modality</span>
              <span className="text-[var(--accent)] font-extrabold uppercase mt-0.5 block">{result.modality}</span>
            </div>
          </div>
        </div>

        {/* Right Confidence Gauge Box (4 cols) */}
        <div className="vs-card lg:col-span-4 p-6 flex flex-col items-center justify-center text-center shadow-md">
          <h4 className="text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-3 font-mono">
            Calibrated Confidence
          </h4>
          <ConfidenceGauge
            confidence={result.confidence}
            verdict={result.verdict}
            size={170}
          />
          <p className="text-[11px] text-[var(--muted)] mt-3">
            Platt-calibrated probability against validation benchmark sets.
          </p>
        </div>

      </div>

      {/* Branch Detection Scores */}
      <div>
        <h3 className="text-base font-bold text-[var(--text)] mb-4 flex items-center gap-2">
          <Layers className="w-4 h-4 text-[var(--accent)]" />
          <span>Branch Detection Scores</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          
          <div className="vs-card p-5 shadow-sm">
            <span className="text-xs font-mono uppercase text-[var(--muted)] block mb-1 font-semibold">Image Artifacts</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-[var(--text)]">
                {result.scores.image !== undefined ? `${Math.round(result.scores.image * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-[var(--text-secondary)] font-mono">CLIP ViT-L/14</span>
            </div>
            <div className="w-full bg-[var(--border)] h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-[var(--accent)] h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.image || 0) * 100}%` }}
              />
            </div>
          </div>

          <div className="vs-card p-5 shadow-sm">
            <span className="text-xs font-mono uppercase text-[var(--muted)] block mb-1 font-semibold">Temporal Video</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-[var(--text)]">
                {result.scores.video !== undefined ? `${Math.round(result.scores.video * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-[var(--text-secondary)] font-mono">Transformer Seq</span>
            </div>
            <div className="w-full bg-[var(--border)] h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-[var(--violet)] h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.video || 0) * 100}%` }}
              />
            </div>
          </div>

          <div className="vs-card p-5 shadow-sm">
            <span className="text-xs font-mono uppercase text-[var(--muted)] block mb-1 font-semibold">Speech Acoustics</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-[var(--text)]">
                {result.scores.audio !== undefined ? `${Math.round(result.scores.audio * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-[var(--text-secondary)] font-mono">Wav2Vec2 XLS-R</span>
            </div>
            <div className="w-full bg-[var(--border)] h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-[var(--warning)] h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.audio || 0) * 100}%` }}
              />
            </div>
          </div>

          <div className="vs-card p-5 shadow-sm">
            <span className="text-xs font-mono uppercase text-[var(--muted)] block mb-1 font-semibold">Lip-Sync Desync</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-[var(--text)]">
                {result.scores.av_sync !== undefined ? `${Math.round(result.scores.av_sync * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-[var(--text-secondary)] font-mono">SyncNet</span>
            </div>
            <div className="w-full bg-[var(--border)] h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-[var(--danger)] h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.av_sync || 0) * 100}%` }}
              />
            </div>
          </div>

        </div>
      </div>

      {/* Visual Explainability Viewer */}
      {result.modality !== 'audio' ? (
        <GradCamViewer
          originalMediaUrl={previewUrl}
          heatmapUrl={result.heatmap_url}
          mediaType={result.modality as 'image' | 'video'}
        />
      ) : (
        <AudioWaveform audioUrl={previewUrl} seekTime={seekTime} />
      )}

      {/* Compact Cryptographic Hash Summary */}
      <div className="vs-card p-5 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center gap-2 text-[var(--muted)] truncate">
          <Hash className="w-4 h-4 text-[var(--accent)] shrink-0" />
          <span className="font-bold text-[var(--text)]">SHA-256:</span>
          <span className="truncate">{result.file_sha256}</span>
        </div>
        <div className="flex items-center gap-2 text-[var(--text-secondary)] shrink-0">
          <Cpu className="w-4 h-4 text-[var(--accent)]" />
          <span>Model: {result.models[0]?.name || 'CLIP ViT-L/14'} (v{result.models[0]?.version || '1.0.0'})</span>
        </div>
      </div>

      {/* Report Modal */}
      {showReportModal && (
        <ReportModal
          result={result}
          filename={originalFile.name}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
};