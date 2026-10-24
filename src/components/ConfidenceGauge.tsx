import React from 'react';
import { motion } from 'motion/react';
import { Verdict } from '../types';

interface ConfidenceGaugeProps {
  confidence: number; // 0 to 1
  verdict: Verdict;
  size?: number;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  verdict,
  size = 200,
}) => {
  const percentage = Math.round(confidence * 100);
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  let colorGradient = {
    start: '#10B981', // Emerald
    end: '#059669',
    glow: 'rgba(16, 185, 129, 0.3)',
    text: 'text-emerald-400',
    border: 'border-emerald-500/30',
    bg: 'bg-emerald-500/10',
    badgeText: 'Authentic Signature',
  };

  if (verdict === 'manipulated') {
    colorGradient = {
      start: '#EF4444', // Red
      end: '#DC2626',
      glow: 'rgba(239, 68, 68, 0.3)',
      text: 'text-rose-400',
      border: 'border-rose-500/30',
      bg: 'bg-rose-500/10',
      badgeText: 'Deepfake Anomalies Detected',
    };
  } else if (verdict === 'inconclusive') {
    colorGradient = {
      start: '#F59E0B', // Amber
      end: '#D97706',
      glow: 'rgba(245, 158, 11, 0.3)',
      text: 'text-amber-400',
      border: 'border-amber-500/30',
      bg: 'bg-amber-500/10',
      badgeText: 'Uncertain Forensic Margin',
    };
  }

  return (
    <div className="flex flex-col items-center justify-center p-4">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-slate-800"
            fill="transparent"
          />

          <defs>
            <linearGradient id={`gauge-grad-${verdict}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={colorGradient.start} />
              <stop offset="100%" stopColor={colorGradient.end} />
            </linearGradient>
            <filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={colorGradient.start} floodOpacity="0.4" />
            </filter>
          </defs>

          {/* Foreground animated meter */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={`url(#gauge-grad-${verdict})`}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeLinecap="round"
            fill="transparent"
            filter="url(#gauge-glow)"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
          />
        </svg>

        {/* Center label */}
        <div className="absolute flex flex-col items-center justify-center text-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className={`text-4xl font-extrabold font-mono tracking-tight ${colorGradient.text}`}
          >
            {percentage}%
          </motion.span>
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-widest mt-1">
            Confidence
          </span>
        </div>
      </div>

      <div className={`mt-3 px-3 py-1 rounded-full text-xs font-semibold ${colorGradient.bg} ${colorGradient.border} border ${colorGradient.text}`}>
        {colorGradient.badgeText}
      </div>
    </div>
  );
};
