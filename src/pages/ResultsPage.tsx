import React, { useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  AlertOctagon,
  HelpCircle,
  Clock,
  Layers,
  FileText,
  Download,
  RotateCcw,
  Cpu,
  ChevronDown,
  ChevronUp,
  Hash,
  ExternalLink,
} from 'lucide-react';
import { motion } from 'motion/react';
import { AnalyzeResponse } from '../types';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { TimelineChart } from '../components/TimelineChart';
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
  const [showTechDetails, setShowTechDetails] = useState(true);
  const [seekTime, setSeekTime] = useState<number | null>(null);

  const verdictConfig = {
    authentic: {
      badgeBg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
      icon: ShieldCheck,
      headline: 'Authentic Media Verified',
      summary: 'No significant synthetic diffusion artifacts, frequency anomalies, or voice cloning patterns detected. Media matches physical camera/microphone signatures.',
    },
    manipulated: {
      badgeBg: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
      icon: AlertOctagon,
      headline: 'Manipulated / Deepfake Content Detected',
      summary: 'Substantial synthetic artifacts detected across spatial attention tokens, spectral frequencies, or temporal facial continuity consistent with generative deepfake models.',
    },
    inconclusive: {
      badgeBg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
      icon: HelpCircle,
      headline: 'Inconclusive Forensic Verdict',
      summary: 'Calibrated certainty falls within the indeterminate margin (40% - 65%). Compression, low resolution, or mixed synthetic-real edits prevent a decisive determination.',
    },
  }[result.verdict];

  const VerdictIcon = verdictConfig.icon;

  const handleSeekFromTimeline = (t: number) => {
    setSeekTime(t);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 sm:py-12 space-y-8">
      {/* Top action header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider font-semibold">
              Inspection Report #{result.id}
            </span>
            <span className="text-slate-600">•</span>
            <span className="text-xs text-slate-400 uppercase font-mono">{result.modality}</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Forensic Analysis Result
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowReportModal(true)}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/30 transition-colors shadow-sm shadow-cyan-500/10"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Forensic Certificate</span>
          </button>
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors border border-slate-700"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Analyze Another</span>
          </button>
        </div>
      </div>

      {/* Main Verdict & Gauge Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Verdict Card */}
        <div className="lg:col-span-2 bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl flex flex-col justify-between">
          <div>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${verdictConfig.badgeBg} mb-4`}>
              <VerdictIcon className="w-4 h-4" />
              <span>Verdict: {result.verdict}</span>
            </div>

            <h3 className="text-2xl sm:text-3xl font-black text-white mb-3">
              {verdictConfig.headline}
            </h3>

            <p className="text-slate-300 text-sm sm:text-base leading-relaxed mb-6">
              {verdictConfig.summary}
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-6 border-t border-slate-800 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[11px]">Duration:</span>
              <span className="text-slate-200 font-bold">{result.processing_ms} ms</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[11px]">File:</span>
              <span className="text-slate-200 font-bold truncate block">{originalFile.name}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[11px]">Size:</span>
              <span className="text-slate-200 font-bold">{(originalFile.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[11px]">Modality:</span>
              <span className="text-cyan-400 font-bold uppercase">{result.modality}</span>
            </div>
          </div>
        </div>

        {/* Confidence Radial Gauge Card */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 shadow-2xl backdrop-blur-xl flex flex-col items-center justify-center text-center">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 font-mono">
            Calibrated Confidence
          </h4>
          <ConfidenceGauge
            confidence={result.confidence}
            verdict={result.verdict}
            size={180}
          />
          <p className="text-[11px] text-slate-500 max-w-xs mt-1">
            Platt temperature-scaled probability against held-out validation sets.
          </p>
        </div>
      </div>

      {/* Warnings Banner if any */}
      {result.warnings && result.warnings.length > 0 && (
        <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200/90 text-xs space-y-1">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Forensic Notice:</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 pl-2 text-slate-300">
            {result.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Per-Modality Branch Scores Grid */}
      <div>
        <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span>Branch Detection Scores</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Image */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
            <span className="text-xs font-mono uppercase text-slate-500 block mb-1">Image Artifacts</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-white">
                {result.scores.image !== undefined ? `${Math.round(result.scores.image * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-slate-400 font-mono">CLIP ViT-L/14</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-cyan-400 h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.image || 0) * 100}%` }}
              />
            </div>
          </div>

          {/* Video */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
            <span className="text-xs font-mono uppercase text-slate-500 block mb-1">Temporal Video</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-white">
                {result.scores.video !== undefined ? `${Math.round(result.scores.video * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-slate-400 font-mono">Transformer Seq</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-violet-400 h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.video || 0) * 100}%` }}
              />
            </div>
          </div>

          {/* Audio */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
            <span className="text-xs font-mono uppercase text-slate-500 block mb-1">Speech Acoustics</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-white">
                {result.scores.audio !== undefined ? `${Math.round(result.scores.audio * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-slate-400 font-mono">Wav2Vec2 XLS-R</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-amber-400 h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.audio || 0) * 100}%` }}
              />
            </div>
          </div>

          {/* AV Sync */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
            <span className="text-xs font-mono uppercase text-slate-500 block mb-1">Lip-Sync Desync</span>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-black font-mono text-white">
                {result.scores.av_sync !== undefined ? `${Math.round(result.scores.av_sync * 100)}%` : 'N/A'}
              </span>
              <span className="text-[11px] text-slate-400 font-mono">SyncNet</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-rose-400 h-full rounded-full transition-all duration-700"
                style={{ width: `${(result.scores.av_sync || 0) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Visual Explainability (Grad-CAM) or Audio Player */}
      {result.modality !== 'audio' ? (
        <GradCamViewer
          originalMediaUrl={previewUrl}
          heatmapUrl={result.heatmap_url}
          mediaType={result.modality as 'image' | 'video'}
        />
      ) : (
        <AudioWaveform audioUrl={previewUrl} seekTime={seekTime} />
      )}

      {/* Temporal Timeline Chart */}
      {result.timeline && result.timeline.length > 0 && (
        <TimelineChart
          timeline={result.timeline}
          onSeek={handleSeekFromTimeline}
          currentTimestamp={seekTime || undefined}
        />
      )}

      {/* Technical Details Accordion */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <button
          onClick={() => setShowTechDetails(!showTechDetails)}
          className="w-full flex items-center justify-between text-left text-sm font-bold text-slate-200"
        >
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>Forensic Metadata & Cryptographic Hashes</span>
          </div>
          {showTechDetails ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {showTechDetails && (
          <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-4 text-xs font-mono">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <span className="text-slate-500 flex items-center gap-1.5">
                <Hash className="w-3.5 h-3.5 text-cyan-400" />
                SHA-256 Digest:
              </span>
              <span className="text-slate-200 break-all select-all">{result.file_sha256}</span>
            </div>

            <div>
              <span className="text-slate-400 font-semibold block mb-2">Evaluated Neural Models:</span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {result.models.map((m, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                    <span className="text-slate-200">{m.name}</span>
                    <span className="text-cyan-400 text-[11px]">v{m.version}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
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
