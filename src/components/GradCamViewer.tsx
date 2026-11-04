
import React, { useState } from 'react';
import {
  Eye,
  EyeOff,
  Layers,
  SlidersHorizontal,
  ScanEye,
  Image as ImageIcon,
} from 'lucide-react';

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

  const panelStyle = {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
  };

  const mediaStyle = {
    background: 'var(--bg-primary)',
  };

  const renderMedia = (className: string) => (
    mediaType === 'image' ? (
      <img
        src={originalMediaUrl}
        alt="Original forensic scan"
        className={className}
      />
    ) : (
      <video
        src={originalMediaUrl}
        controls
        className={className}
      />
    )
  );

  if (!heatmapUrl) {
    return (
      <div
        className="rounded-2xl overflow-hidden p-4"
        style={panelStyle}
      >
        <div
          className="flex items-center gap-2 mb-3 text-xs font-semibold"
          style={{ color: 'var(--text-secondary)' }}
        >
          <ImageIcon size={15} style={{ color: 'var(--accent)' }} />
          Original Media
        </div>

        <div
          className="rounded-xl min-h-[250px] flex justify-center items-center p-3"
          style={mediaStyle}
        >
          {renderMedia('max-h-[460px] max-w-full object-contain rounded-lg')}
        </div>
      </div>
    );
  }

  return (
    <div
      className="w-full rounded-2xl overflow-hidden"
      style={panelStyle}
    >
      {/* HEADER */}
      <div className="p-4 sm:p-5 border-b"
        style={{ borderColor: 'var(--border)' }}>

        <div className="flex flex-wrap items-center justify-between gap-4">

          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{
                background: 'var(--accent-soft)',
                color: 'var(--accent)',
              }}
            >
              <ScanEye size={19} />
            </div>

            <div>
              <h3
                className="text-sm font-bold"
                style={{ color: 'var(--text-primary)' }}
              >
                Visual Attention Analysis
              </h3>

              <p
                className="text-[10px] mt-1"
                style={{ color: 'var(--text-muted)' }}
              >
                GRAD-CAM · {mediaType.toUpperCase()} INSPECTION
              </p>
            </div>
          </div>

          <span
            className="text-[10px] px-2.5 py-1.5 rounded-lg font-bold tracking-wider"
            style={{
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
              border: '1px solid var(--accent-border)',
            }}
          >
            AI ATTENTION MAP
          </span>
        </div>
      </div>

      {/* TOOLBAR */}
      <div
        className="flex flex-wrap items-center justify-between gap-4 p-3 sm:px-5 border-b"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <div
            className="flex p-1 rounded-lg"
            style={{ background: 'var(--bg-secondary)' }}
          >
            <button
              type="button"
              onClick={() => setViewMode('overlay')}
              className="px-3 py-2 rounded-md text-[11px] font-semibold transition-all"
              style={viewMode === 'overlay' ? {
                background: 'var(--accent-soft)',
                color: 'var(--accent)',
              } : {
                color: 'var(--text-muted)',
              }}
            >
              Overlay
            </button>

            <button
              type="button"
              onClick={() => setViewMode('split')}
              className="px-3 py-2 rounded-md text-[11px] font-semibold transition-all"
              style={viewMode === 'split' ? {
                background: 'var(--accent-soft)',
                color: 'var(--accent)',
              } : {
                color: 'var(--text-muted)',
              }}
            >
              Split View
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {viewMode === 'overlay' && showOverlay && (
            <div className="flex items-center gap-2">
              <SlidersHorizontal
                size={14}
                style={{ color: 'var(--text-muted)' }}
              />

              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="w-20 accent-cyan-500 cursor-pointer"
                aria-label="Heatmap opacity"
              />

              <span
                className="font-mono text-[10px] w-8"
                style={{ color: 'var(--accent)' }}
              >
                {Math.round(opacity * 100)}%
              </span>
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowOverlay(!showOverlay)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg border text-[11px] font-semibold transition-all"
            style={{
              background: showOverlay ? 'var(--accent-soft)' : 'var(--surface)',
              borderColor: showOverlay ? 'var(--accent-border)' : 'var(--border)',
              color: showOverlay ? 'var(--accent)' : 'var(--text-secondary)',
            }}
          >
            {showOverlay ? <Eye size={14} /> : <EyeOff size={14} />}
            {showOverlay ? 'CAM ON' : 'CAM OFF'}
          </button>
        </div>
      </div>

      {/* MEDIA VIEW */}
      <div className="p-3 sm:p-5">

        {viewMode === 'overlay' ? (
          <div
            className="relative rounded-xl overflow-hidden flex items-center justify-center min-h-[300px] max-h-[520px] p-2"
            style={mediaStyle}
          >
            {renderMedia('max-h-[500px] max-w-full object-contain rounded-lg')}

            {showOverlay && (
              <img
                src={heatmapUrl}
                alt="Grad-CAM attention heatmap"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-200"
                style={{
                  opacity,
                  mixBlendMode: 'screen',
                }}
              />
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">

            <div
              className="rounded-xl overflow-hidden p-3"
              style={mediaStyle}
            >
              <div className="flex items-center gap-2 mb-3">
                <ImageIcon size={13} style={{ color: 'var(--text-muted)' }} />
                <span
                  className="text-[10px] uppercase tracking-wider font-bold"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  Original Media
                </span>
              </div>

              <div className="min-h-[220px] flex items-center justify-center">
                {renderMedia('max-h-80 max-w-full object-contain rounded-lg')}
              </div>
            </div>

            <div
              className="rounded-xl overflow-hidden p-3"
              style={mediaStyle}
            >
              <div className="flex items-center gap-2 mb-3">
                <Layers size={13} style={{ color: 'var(--accent)' }} />
                <span
                  className="text-[10px] uppercase tracking-wider font-bold"
                  style={{ color: 'var(--accent)' }}
                >
                  Grad-CAM Heatmap
                </span>
              </div>

              <div className="min-h-[220px] flex items-center justify-center">
                {showOverlay ? (
                  <img
                    src={heatmapUrl}
                    alt="Grad-CAM heatmap"
                    className="max-h-80 max-w-full object-contain rounded-lg"
                  />
                ) : (
                  <div
                    className="text-xs"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Heatmap hidden
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* LEGEND */}
      <div
        className="px-4 sm:px-5 py-3 border-t flex flex-wrap items-center justify-between gap-3"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex flex-wrap items-center gap-4 text-[10px]"
          style={{ color: 'var(--text-secondary)' }}>
          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
            High attention
          </span>

          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
            Medium attention
          </span>

          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
            Low attention
          </span>
        </div>

        <span
          className="text-[9px] font-mono"
          style={{ color: 'var(--text-muted)' }}
        >
          VISUAL EXPLANATION · GRAD-CAM
        </span>
      </div>
    </div>
  );
};
