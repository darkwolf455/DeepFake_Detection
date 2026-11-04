
import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div
      className="w-full flex items-center justify-center gap-3 px-4 py-2.5 border-b text-center"
      style={{
        background: 'var(--warning-soft)',
        borderColor: 'var(--warning-border)',
        color: 'var(--text-secondary)',
      }}
    >
      <div
        className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
        style={{ background: 'var(--warning-soft)' }}
      >
        <AlertTriangle
          size={14}
          style={{ color: 'var(--warning)' }}
        />
      </div>

      <p className="text-[11px] sm:text-xs leading-relaxed">
        <strong style={{ color: 'var(--warning)' }}>
          FORENSIC DISCLAIMER
        </strong>
        <span className="mx-2 opacity-40">/</span>
        Detection is probabilistic and can be wrong, especially with
        compressed media or unseen generation methods. Do not use as
        sole evidence.
      </p>
    </div>
  );
};
