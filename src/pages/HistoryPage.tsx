
import React, { useMemo, useState } from 'react';
import {
  History, Trash2, Search, FileImage, FileVideo, FileAudio,
  CheckCircle2, AlertOctagon, HelpCircle, Fingerprint,
  Activity, ShieldCheck, Clock3, SlidersHorizontal, X
} from 'lucide-react';
import { useHistory } from '../context/HistoryContext';

export const HistoryPage: React.FC = () => {
  const { history, deleteHistoryItem, clearHistory } = useHistory();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedModality, setSelectedModality] = useState('all');
  const [selectedVerdict, setSelectedVerdict] = useState('all');

  const filteredHistory = useMemo(() => history.filter((item) => {
    const search = searchTerm.toLowerCase();
    const matchesSearch =
      item.filename.toLowerCase().includes(search) ||
      (item.file_sha256 && item.file_sha256.toLowerCase().includes(search));

    return matchesSearch &&
      (selectedModality === 'all' || item.modality === selectedModality) &&
      (selectedVerdict === 'all' || item.verdict === selectedVerdict);
  }), [history, searchTerm, selectedModality, selectedVerdict]);

  const authentic = history.filter(x => x.verdict === 'authentic').length;
  const manipulated = history.filter(x => x.verdict === 'manipulated').length;
  const inconclusive = history.filter(x => x.verdict === 'inconclusive').length;

  const clearFilters = () => {
    setSearchTerm('');
    setSelectedModality('all');
    setSelectedVerdict('all');
  };

  const getIcon = (modality: string) => {
    if (modality === 'image') return <FileImage className="w-5 h-5 text-cyan-500" />;
    if (modality === 'video') return <FileVideo className="w-5 h-5 text-violet-500" />;
    return <FileAudio className="w-5 h-5 text-amber-500" />;
  };

  const getVerdict = (verdict: string) => {
    if (verdict === 'authentic') return {
      icon: <CheckCircle2 className="w-4 h-4" />,
      style: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20'
    };
    if (verdict === 'manipulated') return {
      icon: <AlertOctagon className="w-4 h-4" />,
      style: 'text-rose-500 bg-rose-500/10 border-rose-500/20'
    };
    return {
      icon: <HelpCircle className="w-4 h-4" />,
      style: 'text-amber-500 bg-amber-500/10 border-amber-500/20'
    };
  };

  return (
    <main className="vs-page min-h-screen px-4 py-9 sm:px-8 lg:px-12">
      <div className="max-w-7xl mx-auto space-y-8">

        <section className="flex flex-col md:flex-row md:items-end justify-between gap-5">
          <div>
            <div className="flex items-center gap-2 text-cyan-500 text-[11px] font-bold tracking-[0.2em] uppercase mb-3">
              <Activity className="w-4 h-4" />
              Intelligence Archive / 001
            </div>
            <h1 className="vs-heading text-3xl sm:text-4xl font-bold tracking-tight">
              Inspection Audit Log<span className="text-cyan-500">.</span>
            </h1>
            <p className="vs-muted text-sm mt-2">
              A chronological record of your multimodal forensic investigations.
            </p>
          </div>

          {history.length > 0 && (
            <button
              onClick={() => {
                if (window.confirm('Clear all forensic history records?')) clearHistory();
              }}
              className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-rose-500/30 text-rose-500 hover:bg-rose-500/10 text-xs font-semibold transition-all"
            >
              <Trash2 size={15} />
              Clear Archive
            </button>
          )}
        </section>

        <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {[
            { label: 'Total Scans', value: history.length, icon: <Fingerprint />, color: 'text-cyan-500' },
            { label: 'Authentic', value: authentic, icon: <ShieldCheck />, color: 'text-emerald-500' },
            { label: 'Manipulated', value: manipulated, icon: <AlertOctagon />, color: 'text-rose-500' },
            { label: 'Inconclusive', value: inconclusive, icon: <HelpCircle />, color: 'text-amber-500' }
          ].map((stat) => (
            <div key={stat.label} className="vs-card p-5 rounded-2xl border transition-all hover:-translate-y-0.5">
              <div className="flex justify-between items-center">
                <span className="vs-muted text-xs">{stat.label}</span>
                <span className={stat.color}>{React.cloneElement(stat.icon, { size: 18 })}</span>
              </div>
              <div className="vs-heading text-3xl font-bold mt-3 tabular-nums">{stat.value}</div>
              <div className="vs-muted text-[10px] mt-1 uppercase tracking-widest">Recorded locally</div>
            </div>
          ))}
        </section>

        <section className="vs-card rounded-2xl p-4 sm:p-5 border space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 vs-heading text-sm font-semibold">
              <SlidersHorizontal size={16} className="text-cyan-500" />
              Refine investigations
            </div>
            {(searchTerm || selectedModality !== 'all' || selectedVerdict !== 'all') && (
              <button onClick={clearFilters} className="text-xs text-cyan-500 flex items-center gap-1">
                <X size={13} /> Reset filters
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="relative">
              <Search size={16} className="absolute left-3 top-3 text-slate-400" />
              <input
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                placeholder="Search filename or SHA-256..."
                className="vs-input w-full rounded-xl pl-10 pr-3 py-3 text-xs outline-none"
              />
            </div>

            <select
              value={selectedModality}
              onChange={e => setSelectedModality(e.target.value)}
              className="vs-input rounded-xl px-3 py-3 text-xs outline-none"
            >
              <option value="all">All modalities</option>
              <option value="image">Images</option>
              <option value="video">Videos</option>
              <option value="audio">Audio</option>
            </select>

            <select
              value={selectedVerdict}
              onChange={e => setSelectedVerdict(e.target.value)}
              className="vs-input rounded-xl px-3 py-3 text-xs outline-none"
            >
              <option value="all">All verdicts</option>
              <option value="authentic">Authentic</option>
              <option value="manipulated">Manipulated</option>
              <option value="inconclusive">Inconclusive</option>
            </select>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="vs-heading font-semibold text-sm">Investigation Records</h3>
            <span className="vs-muted text-xs font-mono">
              {filteredHistory.length} / {history.length} RECORDS
            </span>
          </div>

          {filteredHistory.length === 0 ? (
            <div className="vs-card rounded-3xl border p-12 text-center">
              <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 flex items-center justify-center mx-auto mb-4">
                <History className="w-8 h-8 text-cyan-500" />
              </div>
              <h3 className="vs-heading font-bold text-lg">No investigations found</h3>
              <p className="vs-muted text-sm mt-2 max-w-sm mx-auto">
                {history.length === 0
                  ? 'Your forensic archive is empty. Run your first media analysis to create a record.'
                  : 'No records match your current filters. Try adjusting your search.'}
              </p>
              {history.length > 0 && (
                <button onClick={clearFilters} className="mt-5 text-cyan-500 text-xs font-semibold">
                  Clear all filters
                </button>
              )}
            </div>
          ) : (
            filteredHistory.map(item => {
              const verdict = getVerdict(item.verdict);
              return (
                <article key={item.id} className="vs-card rounded-2xl border p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center justify-between gap-4 transition-all hover:border-cyan-500/40">
                  <div className="flex items-center gap-4 min-w-0">
                    <div className="w-12 h-12 shrink-0 rounded-xl bg-cyan-500/10 border border-cyan-500/10 flex items-center justify-center">
                      {getIcon(item.modality)}
                    </div>
                    <div className="min-w-0">
                      <h4 className="vs-heading text-sm font-bold truncate">{item.filename}</h4>
                      <div className="flex flex-wrap gap-3 vs-muted text-[10px] mt-2 font-mono">
                        <span className="flex items-center gap-1">
                          <Clock3 size={12} /> {item.timestamp}
                        </span>
                        <span className="uppercase">{item.modality}</span>
                        <span className="truncate max-w-40">ID: {item.id}</span>
                      </div>
                      {item.file_sha256 && (
                        <div className="vs-muted text-[10px] mt-2 font-mono truncate max-w-sm">
                          SHA-256: {item.file_sha256}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-between lg:justify-end gap-5">
                    <div className="text-right">
                      <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-[10px] font-bold uppercase tracking-wider ${verdict.style}`}>
                        {verdict.icon}
                        {item.verdict}
                      </div>
                      <div className="vs-heading font-mono font-bold text-sm mt-1.5">
                        {Math.round(item.confidence * 100)}%
                        <span className="vs-muted text-[10px] font-normal ml-1">confidence</span>
                      </div>
                    </div>

                    <button
                      onClick={() => deleteHistoryItem(item.id)}
                      title="Delete record"
                      aria-label={`Delete ${item.filename}`}
                      className="p-2.5 rounded-xl vs-muted hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                    >
                      <Trash2 size={17} />
                    </button>
                  </div>
                </article>
              );
            })
          )}
        </section>
      </div>
    </main>
  );
};

