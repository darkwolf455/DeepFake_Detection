import React, { useState } from 'react';
import {
  History,
  Trash2,
  Search,
  Filter,
  FileImage,
  FileVideo,
  FileAudio,
  CheckCircle2,
  AlertOctagon,
  HelpCircle,
  ExternalLink,
} from 'lucide-react';
import { useHistory } from '../context/HistoryContext';
import { Modality, Verdict } from '../types';

export const HistoryPage: React.FC = () => {
  const { history, deleteHistoryItem, clearHistory } = useHistory();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedModality, setSelectedModality] = useState<string>('all');
  const [selectedVerdict, setSelectedVerdict] = useState<string>('all');

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (item.file_sha256 && item.file_sha256.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesMod = selectedModality === 'all' || item.modality === selectedModality;
    const matchesVer = selectedVerdict === 'all' || item.verdict === selectedVerdict;
    return matchesSearch && matchesMod && matchesVer;
  });

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white flex items-center gap-2.5">
            <History className="w-7 h-7 text-cyan-400" />
            <span>Inspection Audit Log</span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-sm mt-1">
            Local device records of previous forensic scans. Only cryptographic metadata is stored.
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={() => {
              if (window.confirm('Are you sure you want to clear all history records?')) {
                clearHistory();
              }
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 border border-rose-500/30 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Filters Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
          <input
            type="text"
            placeholder="Search filename or SHA-256..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div>
          <select
            value={selectedModality}
            onChange={(e) => setSelectedModality(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Modalities</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="audio">Audio</option>
          </select>
        </div>

        <div>
          <select
            value={selectedVerdict}
            onChange={(e) => setSelectedVerdict(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Verdicts</option>
            <option value="authentic">Authentic</option>
            <option value="manipulated">Manipulated</option>
            <option value="inconclusive">Inconclusive</option>
          </select>
        </div>
      </div>

      {/* Results List */}
      {filteredHistory.length === 0 ? (
        <div className="p-12 text-center rounded-3xl bg-slate-900/40 border border-slate-800/80">
          <History className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <h4 className="text-base font-bold text-slate-300">No matching audit records</h4>
          <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
            {history.length === 0
              ? 'Run an analysis scan in the Analyze tab to generate forensic logs.'
              : 'Try clearing your search query or selecting "All".'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredHistory.map((item) => {
            const isManipulated = item.verdict === 'manipulated';
            const isAuthentic = item.verdict === 'authentic';

            return (
              <div
                key={item.id}
                className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0">
                    {item.modality === 'image' && <FileImage className="w-5 h-5 text-cyan-400" />}
                    {item.modality === 'video' && <FileVideo className="w-5 h-5 text-violet-400" />}
                    {item.modality === 'audio' && <FileAudio className="w-5 h-5 text-amber-400" />}
                  </div>

                  <div className="space-y-0.5 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-white truncate max-w-xs sm:max-w-md">
                        {item.filename}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                        #{item.id}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500 font-mono">
                      <span>{item.timestamp}</span>
                      <span>•</span>
                      <span className="truncate max-w-[200px]">{item.file_sha256}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between sm:justify-end gap-4 shrink-0">
                  <div className="text-right">
                    <div className="flex items-center gap-1.5">
                      {isManipulated && <AlertOctagon className="w-3.5 h-3.5 text-rose-400" />}
                      {isAuthentic && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
                      {!isManipulated && !isAuthentic && <HelpCircle className="w-3.5 h-3.5 text-amber-400" />}
                      <span
                        className={`text-xs font-bold uppercase font-mono ${
                          isManipulated
                            ? 'text-rose-400'
                            : isAuthentic
                            ? 'text-emerald-400'
                            : 'text-amber-400'
                        }`}
                      >
                        {item.verdict}
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-400 font-mono block">
                      {Math.round(item.confidence * 100)}% Confidence
                    </span>
                  </div>

                  <button
                    onClick={() => deleteHistoryItem(item.id)}
                    className="p-2 text-slate-500 hover:text-rose-400 rounded-lg hover:bg-slate-800 transition-colors"
                    title="Delete Record"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
