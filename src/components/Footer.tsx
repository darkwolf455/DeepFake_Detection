import { ArrowUpRight, Fingerprint } from 'lucide-react';

export function Footer({ onViewModels }: { onViewModels: () => void }) {
  return (
    <footer className="page-footer">
      <span className="actions">
        <Fingerprint size={14} /> VeriSense · Student research project
      </span>
      <button
        className="actions"
        onClick={onViewModels}
        style={{
          background: 'none',
          border: 0,
          color: 'inherit',
          fontSize: 10,
        }}
      >
        Explore the models <ArrowUpRight size={13} />
      </button>
    </footer>
  );
}
