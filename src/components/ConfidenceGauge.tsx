
import React from 'react';
import { motion } from 'motion/react';
import { Verdict } from '../types';

interface ConfidenceGaugeProps {
  confidence: number;
  verdict: Verdict;
  size?: number;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  verdict,
  size = 200,
}) => {
  const percentage = Math.max(0, Math.min(100, Math.round(confidence * 100)));

  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const strokeDashoffset =
    circumference - (percentage / 100) * circumference;

  const config = {
    authentic: {
      color: '#10B981',
      glow: 'rgba(16,185,129,0.25)',
      label: 'Authentic Signature',
      status: 'AUTHENTIC',
    },
    manipulated: {
      color: '#F43F5E',
      glow: 'rgba(244,63,94,0.25)',
      label: 'Deepfake Anomalies Detected',
      status: 'MANIPULATED',
    },
    inconclusive: {
      color: '#F59E0B',
      glow: 'rgba(245,158,11,0.25)',
      label: 'Uncertain Forensic Margin',
      status: 'INCONCLUSIVE',
    },
  };

  const current =
    config[verdict] ?? config.inconclusive;

  return (
    <div
      className="w-full h-full flex flex-col items-center justify-center rounded-2xl p-5"
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
      }}
    >
      {/* HEADER */}
      <div className="w-full flex items-center justify-between mb-5">
        <div>
          <p
            className="text-[10px] font-bold tracking-[0.16em] uppercase"
            style={{ color: 'var(--text-muted)' }}
          >
            Forensic Analysis
          </p>

          <h3
            className="text-sm font-bold mt-1"
            style={{ color: 'var(--text-primary)' }}
          >
            Confidence Score
          </h3>
        </div>

        <span
          className="w-2 h-2 rounded-full animate-pulse"
          style={{ background: current.color }}
        />
      </div>

      {/* GAUGE */}
      <div
        className="relative flex items-center justify-center"
        style={{ width: size, height: size }}
      >
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="-rotate-90 overflow-visible"
        >
          <defs>
            <filter id="verisense-gauge-glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="var(--border)"
            strokeWidth={strokeWidth}
            fill="none"
          />

          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={current.color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeLinecap="round"
            fill="none"
            filter="url(#verisense-gauge-glow)"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.4, ease: 'easeOut' }}
          />
        </svg>

        {/* CENTER TEXT */}
        <div className="absolute flex flex-col items-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-[42px] font-extrabold tracking-tight font-mono"
            style={{
              color: current.color,
              textShadow: `0 0 24px ${current.glow}`,
            }}
          >
            {percentage}%
          </motion.span>

          <span
            className="text-[10px] uppercase tracking-[0.18em] font-bold mt-1"
            style={{ color: 'var(--text-muted)' }}
          >
            Confidence
          </span>
        </div>
      </div>

      {/* VERDICT */}
      <div
        className="mt-5 w-full text-center py-3 px-3 rounded-xl border"
        style={{
          background: `${current.color}12`,
          borderColor: `${current.color}45`,
        }}
      >
        <p
          className="text-[10px] font-extrabold tracking-[0.15em]"
          style={{ color: current.color }}
        >
          {current.status}
        </p>

        <p
          className="text-[11px] mt-1"
          style={{ color: 'var(--text-secondary)' }}
        >
          {current.label}
        </p>
      </div>

      <p
        className="text-[10px] mt-3 text-center leading-relaxed"
        style={{ color: 'var(--text-muted)' }}
      >
        Model-estimated confidence. Not a probability of factual authenticity.
      </p>
    </div>
  );
};
