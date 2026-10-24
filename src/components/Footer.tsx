import React from 'react';
import { ShieldCheck, Cpu, GitBranch, Lock } from 'lucide-react';

export const Footer: React.FC<{ onNavigateToCode?: () => void }> = ({ onNavigateToCode }) => {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-[#0B0F19] text-slate-400 text-xs py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="md:col-span-2 space-y-2">
            <div className="flex items-center gap-2 text-slate-200 font-bold text-sm">
              <ShieldCheck className="w-4 h-4 text-cyan-400" />
              <span>VeriSense Multimodal Forensic Platform</span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed max-w-md">
              A multimodal deepfake detection system detecting synthetic artifacts in images, video sequences, and audio waveforms using CLIP ViT-L/14, temporal sequence transformers, and Wav2Vec2 XLS-R.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="text-slate-200 font-semibold text-xs tracking-wider uppercase">Architecture</h4>
            <ul className="space-y-1 text-slate-400">
              <li>Vision: Frozen CLIP ViT-L/14 + Head</li>
              <li>Temporal: 4-Layer Transformer Encoder</li>
              <li>Audio: Wav2Vec2-XLS-R-300M + Attn Pooling</li>
              <li>Fusion: Platt Calibrated Late-Fusion MLP</li>
            </ul>
          </div>

          <div className="space-y-2">
            <h4 className="text-slate-200 font-semibold text-xs tracking-wider uppercase">Privacy & Safety</h4>
            <div className="flex items-center gap-2 text-slate-400">
              <Lock className="w-3.5 h-3.5 text-cyan-400" />
              <span>Zero media retention (in-memory unlinking)</span>
            </div>
            <div className="flex items-center gap-2 text-slate-400">
              <Cpu className="w-3.5 h-3.5 text-violet-400" />
              <span>GPU Accelerated (FP16 AMP)</span>
            </div>
            {onNavigateToCode && (
              <button
                onClick={onNavigateToCode}
                className="inline-flex items-center gap-1.5 text-cyan-400 hover:text-cyan-300 mt-2 font-mono text-[11px]"
              >
                <GitBranch className="w-3 h-3" />
                Browse Python Source & Weights
              </button>
            )}
          </div>
        </div>

        <div className="pt-6 border-t border-slate-800/60 flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-500 text-[11px]">
          <div>
            &copy; 2026 VeriSense Research Project. All algorithms calibrated on FaceForensics++, GenImage & ASVspoof 2019.
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            SHA-256 Verified • RFC 4122 UUID v4
          </div>
        </div>
      </div>
    </footer>
  );
};
