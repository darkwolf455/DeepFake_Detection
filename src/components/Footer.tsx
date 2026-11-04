
import React from 'react';
import {
  ShieldCheck,
  Cpu,
  GitBranch,
  Lock,
  ArrowUpRight,
  Fingerprint,
} from 'lucide-react';

export const Footer: React.FC<{
  onNavigateToCode?: () => void;
}> = ({ onNavigateToCode }) => {
  return (
    <footer
      className="w-full mt-auto border-t"
      style={{
        background: 'var(--footer-bg)',
        borderColor: 'var(--border)',
        color: 'var(--text-secondary)',
      }}
    >
      <div className="max-w-[1500px] mx-auto px-5 sm:px-8 lg:px-10 py-10">

        {/* TOP SECTION */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-9 pb-9">

          {/* BRAND */}
          <div className="lg:col-span-2 space-y-4">

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-cyan-400 to-violet-600 shadow-lg shadow-cyan-500/15">
                <ShieldCheck className="w-5 h-5 text-white" />
              </div>

              <div>
                <h3
                  className="font-extrabold text-base tracking-tight"
                  style={{ color: 'var(--text-primary)' }}
                >
                  VeriSense
                </h3>
                <p className="text-[10px] tracking-[0.16em] font-semibold"
                  style={{ color: 'var(--text-muted)' }}>
                  MULTIMODAL FORENSICS
                </p>
              </div>
            </div>

            <p className="text-xs leading-6 max-w-md"
              style={{ color: 'var(--text-secondary)' }}>
              An intelligent multimodal deepfake forensic platform
              designed to identify synthetic artifacts across images,
              video sequences and audio waveforms.
            </p>

            <div
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border text-[10px] font-semibold tracking-wide"
              style={{
                background: 'var(--surface)',
                borderColor: 'var(--border)',
                color: 'var(--text-secondary)',
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              RESEARCH PLATFORM · VERSION 1.0
            </div>
          </div>

          {/* ARCHITECTURE */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Cpu size={15} style={{ color: 'var(--accent)' }} />
              <h4
                className="text-[11px] font-bold tracking-[0.14em] uppercase"
                style={{ color: 'var(--text-primary)' }}
              >
                Architecture
              </h4>
            </div>

            <ul className="space-y-3 text-[11px] leading-relaxed">
              <li>
                <span style={{ color: 'var(--text-primary)' }}>Vision</span>
                <p>CLIP ViT-L/14 + Classification Head</p>
              </li>
              <li>
                <span style={{ color: 'var(--text-primary)' }}>Temporal</span>
                <p>4-Layer Transformer Encoder</p>
              </li>
              <li>
                <span style={{ color: 'var(--text-primary)' }}>Audio</span>
                <p>Wav2Vec2 XLS-R + Attention Pooling</p>
              </li>
              <li>
                <span style={{ color: 'var(--text-primary)' }}>Fusion</span>
                <p>Calibrated Late-Fusion MLP</p>
              </li>
            </ul>
          </div>

          {/* SECURITY */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Lock size={15} style={{ color: 'var(--accent)' }} />
              <h4
                className="text-[11px] font-bold tracking-[0.14em] uppercase"
                style={{ color: 'var(--text-primary)' }}
              >
                Security & Privacy
              </h4>
            </div>

            <div className="space-y-3 text-[11px]">
              <div className="flex items-start gap-2.5">
                <Lock size={14} className="mt-0.5 shrink-0 text-emerald-500" />
                <span>Zero media retention (in-memory unlinking)</span>
              </div>

              <div className="flex items-start gap-2.5">
                <Cpu size={14} className="mt-0.5 shrink-0 text-violet-500" />
                <span>GPU Accelerated (FP16 AMP)</span>
              </div>

              <div className="flex items-start gap-2.5">
                <Fingerprint size={14} className="mt-0.5 shrink-0"
                  style={{ color: 'var(--accent)' }} />
                <span>SHA-256 forensic integrity verification</span>
              </div>
            </div>

            {onNavigateToCode && (
              <button
                type="button"
                onClick={onNavigateToCode}
                className="inline-flex items-center gap-2 text-[11px] font-semibold transition-colors hover:opacity-70"
                style={{ color: 'var(--accent)' }}
              >
                <GitBranch size={13} />
                Explore Source Code
                <ArrowUpRight size={12} />
              </button>
            )}
          </div>
        </div>

        {/* BOTTOM SECTION */}
        <div
          className="pt-5 border-t flex flex-col md:flex-row items-start md:items-center justify-between gap-3"
          style={{ borderColor: 'var(--border)' }}
        >
          <p className="text-[10px] leading-relaxed"
            style={{ color: 'var(--text-muted)' }}>
            © 2026 VeriSense Research Project. All rights reserved.
          </p>

          <div
            className="flex flex-wrap items-center gap-3 text-[10px] font-mono"
            style={{ color: 'var(--text-muted)' }}
          >
            <span>SHA-256 VERIFIED</span>
            <span className="opacity-30">/</span>
            <span>RFC 4122 UUID V4</span>
            <span className="opacity-30">/</span>
            <span>VERISENSE V1.0</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
