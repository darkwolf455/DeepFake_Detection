import React from 'react';
import { X, Download, Printer, ShieldCheck, CheckCircle2, AlertOctagon, HelpCircle } from 'lucide-react';
import { AnalyzeResponse } from '../types';

interface ReportModalProps {
  result: AnalyzeResponse;
  filename: string;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({ result, filename, onClose }) => {
  const handlePrint = () => {
    window.print();
  };

  const handleDownloadJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `verisense_forensic_report_${result.id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const verdictColor =
    result.verdict === 'manipulated'
      ? 'text-rose-500 border-rose-500/30 bg-rose-500/10'
      : result.verdict === 'authentic'
      ? 'text-emerald-500 border-emerald-500/30 bg-emerald-500/10'
      : 'text-amber-500 border-amber-500/30 bg-amber-500/10';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl text-slate-100">
        {/* Top actions */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-cyan-400" />
            <h2 className="text-lg font-bold">VeriSense Forensic Inspection Certificate</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print / Save PDF</span>
            </button>
            <button
              onClick={handleDownloadJson}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export JSON</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Certificate Body */}
        <div className="mt-6 space-y-6 text-sm" id="printable-certificate">
          {/* Header details */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-2xl bg-slate-950/60 border border-slate-800 text-xs font-mono">
            <div>
              <span className="text-slate-500 block">Report ID:</span>
              <span className="text-cyan-400 font-bold">{result.id}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Modality:</span>
              <span className="text-slate-200 uppercase">{result.modality}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Inspection Date:</span>
              <span className="text-slate-200">{new Date().toISOString().slice(0, 10)}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Processing Time:</span>
              <span className="text-slate-200">{result.processing_ms} ms</span>
            </div>
          </div>

          {/* Verdict Banner */}
          <div className={`p-6 rounded-2xl border flex flex-col sm:flex-row items-center justify-between gap-4 ${verdictColor}`}>
            <div className="flex items-center gap-4">
              {result.verdict === 'manipulated' && <AlertOctagon className="w-10 h-10" />}
              {result.verdict === 'authentic' && <CheckCircle2 className="w-10 h-10" />}
              {result.verdict === 'inconclusive' && <HelpCircle className="w-10 h-10" />}
              <div>
                <span className="text-xs font-mono tracking-wider uppercase block opacity-80">Calibrated Verdict</span>
                <span className="text-2xl font-black uppercase tracking-wide">{result.verdict}</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs font-mono block opacity-80">Confidence Level</span>
              <span className="text-3xl font-black font-mono">{Math.round(result.confidence * 100)}%</span>
            </div>
          </div>

          {/* Target File Hash */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 font-mono">
              Evidence Cryptographic Integrity
            </h4>
            <div className="space-y-1 font-mono text-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <span className="text-slate-500">File Name:</span>
                <span className="text-slate-200 break-all">{filename}</span>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <span className="text-slate-500">SHA-256 Digest:</span>
                <span className="text-cyan-400 break-all text-[11px]">{result.file_sha256 || 'Not Available'}</span>
              </div>
            </div>
          </div>

          {/* Branch Scores */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Per-Modality Forensic Breakdown
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              {Object.entries(result.scores).map(([mod, score]) => (
                <div key={mod} className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block uppercase font-mono text-[10px]">{mod.replace('_', ' ')}</span>
                  <span className="text-base font-bold font-mono text-slate-100">
                    {score !== undefined && score !== null ? `${Math.round(score * 100)}%` : 'N/A'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Active Model Stack */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Neural Network Pipeline Stack
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {result.models.map((m, i) => (
                <li key={i} className="flex items-center justify-between p-2 rounded-lg bg-slate-950/50 border border-slate-800/80">
                  <span className="font-medium">{m.name}</span>
                  <span className="font-mono text-cyan-400 text-[11px]">v{m.version}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal disclaimer in report */}
          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200/90 text-xs leading-relaxed">
            <strong>Mandatory Notice:</strong> Detection is probabilistic and can be wrong, especially for compressed media or unseen generation methods. Do not use as sole evidence in legal, punitive, or defamation proceedings.
          </div>
        </div>
      </div>
    </div>
  );
};
