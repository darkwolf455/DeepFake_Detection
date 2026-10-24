import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div className="w-full bg-amber-500/10 border-b border-amber-500/20 text-amber-200/90 px-4 py-2 text-xs flex items-center justify-center gap-2">
      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
      <span className="text-center font-medium">
        <strong>Forensic Disclaimer:</strong> Detection is probabilistic and can be wrong, especially for compressed media or unseen generation methods. Do not use as sole evidence.
      </span>
    </div>
  );
};
