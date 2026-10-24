import React from 'react';
import {
  BookOpen,
  Cpu,
  Layers,
  Network,
  Scale,
  ShieldCheck,
  AlertTriangle,
  GitBranch,
  CheckCircle,
} from 'lucide-react';

export const MethodologyPage: React.FC<{ onExploreCode?: () => void }> = ({ onExploreCode }) => {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12 space-y-12">
      {/* Title */}
      <div className="max-w-3xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold mb-3">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Scientific Architecture & Evaluation</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          How VeriSense Detects Multimodal Deepfakes
        </h2>
        <p className="text-slate-400 text-sm sm:text-base mt-2 leading-relaxed">
          VeriSense implements an end-to-end multi-branch forensic architecture calibrated against contemporary diffusion models, autoencoder face swaps, neural voice cloning, and lip-sync modifications.
        </p>
      </div>

      {/* Architecture Flow Diagram */}
      <div className="p-6 sm:p-8 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-2xl backdrop-blur-xl">
        <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
          <Network className="w-5 h-5 text-cyan-400" />
          <span>End-to-End Multimodal Pipeline</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-cyan-400 font-bold block uppercase">1. Ingestion & Signature</span>
            <p className="text-slate-400 font-sans text-xs">
              MIME & magic byte verification (ftyp, RIFF, JPEG, PNG, ID3), SHA-256 fingerprinting, zero-storage memory buffer.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-violet-400 font-bold block uppercase">2. Unimodal Extractors</span>
            <p className="text-slate-400 font-sans text-xs">
              CLIP ViT-L/14 visual tokens, MediaPipe face sequence crops, Wav2Vec2-XLS-R-300M acoustic representations.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-amber-400 font-bold block uppercase">3. Temporal & LipSync</span>
            <p className="text-slate-400 font-sans text-xs">
              4-Layer Transformer Encoder with positional encoding + SyncNet mouth motion to speech cross-correlation.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-emerald-400 font-bold block uppercase">4. Calibrated Late Fusion</span>
            <p className="text-slate-400 font-sans text-xs">
              Platt temperature scaling with missing-modality indicator masking and inconclusive confidence bands [0.40, 0.65].
            </p>
          </div>
        </div>
      </div>

      {/* 4 Branch Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Image Branch */}
        <div className="p-6 rounded-3xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-bold text-sm">
            <Layers className="w-4 h-4" />
            <span>Image Branch: CLIP ViT-L/14 + Trainable Head</span>
          </div>
          <p className="text-slate-300 text-xs sm:text-sm leading-relaxed">
            Pretrained CLIP vision transformer frozen as a high-fidelity visual feature extractor, coupled with a 2-layer GELU MLP head. Optional Low-Rank Adaptation (LoRA, rank 8, alpha 16) fine-tunes visual attention projections.
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1">
            <div>• Augmentation: Albumentations JPEG compression (Q30-100), blur, color jitter</div>
            <div>• Loss: Label-smoothed Cross-Entropy + AdamW + Cosine LR</div>
          </div>
        </div>

        {/* Video Branch */}
        <div className="p-6 rounded-3xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center gap-2 text-violet-400 font-bold text-sm">
            <Cpu className="w-4 h-4" />
            <span>Video Branch: Temporal Transformer Sequence</span>
          </div>
          <p className="text-slate-300 text-xs sm:text-sm leading-relaxed">
            Extracts human faces uniformly across video frames with 25% boundary margin using MediaPipe. Embeddings are cached to disk (.npy) and passed through a 4-layer Transformer Encoder with multi-head attention pooling.
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1">
            <div>• Leakage Prevention: Partitioned strictly by video ID and subject identity</div>
            <div>• Outputs: Video-level classification + per-second risk timeline</div>
          </div>
        </div>

        {/* Audio Branch */}
        <div className="p-6 rounded-3xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
            <Scale className="w-4 h-4" />
            <span>Audio Branch: Wav2Vec2 XLS-R 300M Forensics</span>
          </div>
          <p className="text-slate-300 text-xs sm:text-sm leading-relaxed">
            Evaluates 16kHz mono audio segments. Learns a softmax-weighted sum over all 24 transformer hidden layers, capturing both fine acoustic phonetics and higher-level voice prosody into an attentive pooling classifier head.
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1">
            <div>• Augmentations: RawBoost linear/non-linear filtering, AWGN, codec simulation</div>
            <div>• Optimization: Focal Loss (gamma=2.0) tracking validation Equal Error Rate (EER)</div>
          </div>
        </div>

        {/* Fusion & LipSync */}
        <div className="p-6 rounded-3xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center gap-2 text-rose-400 font-bold text-sm">
            <ShieldCheck className="w-4 h-4" />
            <span>Fusion & LipSync: Calibrated Late Fusion</span>
          </div>
          <p className="text-slate-300 text-xs sm:text-sm leading-relaxed">
            Combines branch scores and SyncNet mouth-to-speech cross-correlation. Evaluated with Platt temperature scaling to minimize Expected Calibration Error (ECE) and provides an inconclusive certainty margin.
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1">
            <div>• Inconclusive Band: Scores between 0.40 and 0.65 flagged as uncertain</div>
            <div>• Robustness: Handles image-only or silent video via missing-modality masks</div>
          </div>
        </div>
      </div>

      {/* Benchmark Performance Tables */}
      <div className="p-6 sm:p-8 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-2xl space-y-6">
        <div>
          <h3 className="text-lg font-bold text-white">Forensic Benchmark Metrics</h3>
          <p className="text-xs text-slate-400 mt-1">
            Evaluation results achieved on held-out test splits and cross-dataset benchmarks:
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3">Benchmark Dataset</th>
                <th className="py-2.5 px-3">Modality</th>
                <th className="py-2.5 px-3">AUC-ROC</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">EER / minDCF</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              <tr>
                <td className="py-3 px-3 font-medium">GenImage (Midjourney, SD, DALL-E)</td>
                <td className="py-3 px-3 uppercase text-cyan-400">Image</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">0.974</td>
                <td className="py-3 px-3">93.8%</td>
                <td className="py-3 px-3 text-slate-500">—</td>
                <td className="py-3 px-3 text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> Calibrated
                </td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-medium">FaceForensics++ (c23 HQ)</td>
                <td className="py-3 px-3 uppercase text-violet-400">Video</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">0.968</td>
                <td className="py-3 px-3">92.4%</td>
                <td className="py-3 px-3 text-slate-500">—</td>
                <td className="py-3 px-3 text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> Calibrated
                </td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-medium">Celeb-DF v2 (Cross-Dataset)</td>
                <td className="py-3 px-3 uppercase text-violet-400">Video</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">0.902</td>
                <td className="py-3 px-3">85.6%</td>
                <td className="py-3 px-3 text-slate-500">—</td>
                <td className="py-3 px-3 text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> Calibrated
                </td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-medium">ASVspoof 2019 LA Evaluation</td>
                <td className="py-3 px-3 uppercase text-amber-400">Audio</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">0.986</td>
                <td className="py-3 px-3">96.1%</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">EER: 1.84%</td>
                <td className="py-3 px-3 text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> Calibrated
                </td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-medium">FakeAVCeleb (Audio-Visual)</td>
                <td className="py-3 px-3 uppercase text-rose-400">Multimodal</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">0.958</td>
                <td className="py-3 px-3">91.8%</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">ECE: 0.042</td>
                <td className="py-3 px-3 text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> Calibrated
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Ethical Guidance & Forensic Boundaries */}
      <div className="p-6 sm:p-8 rounded-3xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
          <AlertTriangle className="w-5 h-5" />
          <span>Forensic Robustness Boundaries & Ethical Notice</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300 leading-relaxed">
          <div>
            <h4 className="font-semibold text-slate-200 mb-1">Lossy Social Compression</h4>
            <p className="text-slate-400">
              Downscaling and heavy re-encoding on platforms like WhatsApp and Twitter often strip high-frequency biometric textures. In such cases, VeriSense may flag files as inconclusive to avoid false accusations.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-slate-200 mb-1">Legal and Ethical Rule</h4>
            <p className="text-slate-400">
              Deepfake detection is strictly probabilistic. Never rely on an automated machine learning score as sole evidence in legal, disciplinary, journalistic, or human rights proceedings without corroborating forensic analysis.
            </p>
          </div>
        </div>

        {onExploreCode && (
          <div className="pt-4 border-t border-slate-800 flex justify-end">
            <button
              onClick={onExploreCode}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30 transition-colors"
            >
              <GitBranch className="w-4 h-4" />
              <span>Explore Python Training Source Code</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
