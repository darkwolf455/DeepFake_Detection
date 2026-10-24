import React, { useState } from 'react';
import { Eye, EyeOff, Layers, Sliders, Maximize2 } from 'lucide-react';

interface GradCamViewerProps {
  originalMediaUrl: string;
  heatmapUrl?: string;
  mediaType: 'image' | 'video';
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({
  originalMediaUrl,
  heatmapUrl,
  mediaType,
}) => {
  const [opacity, setOpacity] = useState(0.65);
  const [showOverlay, setShowOverlay] = useState(true);
  const [viewMode, setViewMode] = useState<'overlay' | 'split'>('overlay');

  if (!heatmapUrl) {
    return (
      <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center p-8 text-center">
        {mediaType === 'image' ? (
          <img
            src={originalMediaUrl}
            alt="Original forensic scan"
            className="max-h-96 object-contain rounded-xl"
          />
        ) : (
          <video
            src={originalMediaUrl}
            controls
            className="max-h-96 object-contain rounded-xl"
          />
        )}
      </div>
    );
  }

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3 pb-3 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-slate-200">
            Grad-CAM Visual Attention Saliency
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Opacity slider */}
          {showOverlay && viewMode === 'overlay' && (
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400 text-[11px]">Heatmap Opacity:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-24 accent-cyan-400 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
              />
              <span className="font-mono text-[11px] text-cyan-300 w-8">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          )}

          {/* View mode toggle */}
          <div className="flex items-center rounded-lg bg-slate-800 p-0.5 border border-slate-700">
            <button
              onClick={() => setViewMode('overlay')}
              className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                viewMode === 'overlay'
                  ? 'bg-cyan-500/20 text-cyan-300'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Overlay
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                viewMode === 'split'
                  ? 'bg-cyan-500/20 text-cyan-300'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Split View
            </button>
          </div>

          {/* Toggle visibility */}
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border transition-colors ${
              showOverlay
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>{showOverlay ? 'Hide CAM' : 'Show CAM'}</span>
          </button>
        </div>
      </div>

      {/* Media Canvas View */}
      {viewMode === 'overlay' ? (
        <div className="relative rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center min-h-[300px] max-h-[460px]">
          {mediaType === 'image' ? (
            <img
              src={originalMediaUrl}
              alt="Scan"
              className="max-h-[460px] w-auto object-contain rounded-lg"
            />
          ) : (
            <video
              src={originalMediaUrl}
              controls
              className="max-h-[460px] w-auto object-contain rounded-lg"
            />
          )}

          {/* Grad-CAM Heatmap Overlay */}
          {showOverlay && (
            <img
              src={heatmapUrl}
              alt="Grad-CAM Saliency Map"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen transition-opacity duration-150"
              style={{ opacity }}
            />
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="relative rounded-xl overflow-hidden bg-slate-950 flex flex-col items-center justify-center p-2 border border-slate-800">
            <span className="text-[11px] text-slate-400 mb-1 font-mono">Original Media</span>
            {mediaType === 'image' ? (
              <img src={originalMediaUrl} alt="Original" className="max-h-72 object-contain rounded" />
            ) : (
              <video src={originalMediaUrl} controls className="max-h-72 object-contain rounded" />
            )}
          </div>
          <div className="relative rounded-xl overflow-hidden bg-slate-950 flex flex-col items-center justify-center p-2 border border-slate-800">
            <span className="text-[11px] text-cyan-400 mb-1 font-mono">ViT Attention Saliency (Grad-CAM)</span>
            <img src={heatmapUrl} alt="Heatmap" className="max-h-72 object-contain rounded" />
          </div>
        </div>
      )}

      <div className="flex items-center justify-between mt-3 text-[11px] text-slate-500 font-mono">
        <span>Red/Yellow: High manipulation saliency (blend boundaries, warped textures)</span>
        <span>Blue: Low artifact concentration</span>
      </div>
    </div>
  );
};
