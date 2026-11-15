import { useMemo, useState } from 'react';
import {
  AudioLines,
  FileImage,
  History,
  Search,
  Trash2,
  Video,
} from 'lucide-react';
import { useHistory } from '../context/HistoryContext';

export function HistoryPage() {
  const { history, deleteHistoryItem, clearHistory } = useHistory();
  const [search, setSearch] = useState('');
  const [modality, setModality] = useState('all');
  const [verdict, setVerdict] = useState('all');
  const filtered = useMemo(
    () =>
      history.filter(
        (item) =>
          `${item.filename} ${item.file_sha256 ?? ''}`
            .toLowerCase()
            .includes(search.toLowerCase()) &&
          (modality === 'all' || item.modality === modality) &&
          (verdict === 'all' || item.verdict === verdict),
      ),
    [history, search, modality, verdict],
  );
  const reset = () => {
    setSearch('');
    setModality('all');
    setVerdict('all');
  };
  return (
    <div className="page stack">
      <section className="page-heading">
        <div className="eyebrow">
          <History size={13} /> Your recent investigations
        </div>
        <h1>A record of every closer look.</h1>
        <p>
          The latest 50 results are saved in this browser. Uploaded media files
          are not retained.
        </p>
      </section>
      <div className="score-grid">
        {[
          { label: 'Total analyses', value: history.length },
          {
            label: 'Not flagged',
            value: history.filter((item) => item.verdict === 'authentic')
              .length,
          },
          {
            label: 'Flagged',
            value: history.filter((item) => item.verdict === 'manipulated')
              .length,
          },
          {
            label: 'Inconclusive',
            value: history.filter((item) => item.verdict === 'inconclusive')
              .length,
          },
        ].map((stat) => (
          <div className="panel score-card" key={stat.label}>
            <span className="small muted">{stat.label}</span>
            <div className="score-value">{stat.value}</div>
          </div>
        ))}
      </div>
      <section className="panel">
        <div className="panel-heading">
          <h2 className="section-title">
            <Search /> Find a result
          </h2>
          <button className="secondary-button" onClick={reset}>
            Reset filters
          </button>
        </div>
        <div className="history-filters">
          <input
            className="text-input"
            aria-label="Search history"
            placeholder="Search filename or SHA-256…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <select
            className="text-input"
            aria-label="Filter by modality"
            value={modality}
            onChange={(event) => setModality(event.target.value)}
          >
            <option value="all">All media</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="audio">Audio</option>
          </select>
          <select
            className="text-input"
            aria-label="Filter by verdict"
            value={verdict}
            onChange={(event) => setVerdict(event.target.value)}
          >
            <option value="all">All decisions</option>
            <option value="authentic">Not flagged</option>
            <option value="manipulated">Flagged</option>
            <option value="inconclusive">Inconclusive</option>
          </select>
        </div>
      </section>
      <div className="actions" style={{ justifyContent: 'space-between' }}>
        <span className="small muted">
          {filtered.length} of {history.length} records
        </span>
        {!!history.length && (
          <button
            className="secondary-button"
            onClick={() => {
              if (
                window.confirm(
                  'Clear all saved analysis history in this browser?',
                )
              )
                clearHistory();
            }}
          >
            <Trash2 size={13} />
            Clear history
          </button>
        )}
      </div>
      {!filtered.length ? (
        <div className="panel empty-state">
          <History />
          <h2>{history.length ? 'No matching results.' : 'A clean slate.'}</h2>
          <p>
            {history.length
              ? 'Try another filename or reset your filters.'
              : 'Analyze your first image, video or speech clip to start your history.'}
          </p>
        </div>
      ) : (
        filtered.map((item) => {
          const Icon =
            item.modality === 'image'
              ? FileImage
              : item.modality === 'video'
                ? Video
                : AudioLines;
          const timestamp = new Date(item.timestamp);
          return (
            <article className="panel history-row" key={item.id}>
              <div className="file-heading">
                <Icon />
                <div>
                  <h3>{item.filename}</h3>
                  <span>
                    {Number.isNaN(timestamp.getTime())
                      ? item.timestamp
                      : timestamp.toLocaleString()}{' '}
                    ·{' '}
                    {item.scores.fusion !== undefined
                      ? 'VIDEO + AUDIO'
                      : item.modality.toUpperCase()}
                  </span>
                </div>
              </div>
              <div className="actions" style={{ flexShrink: 0 }}>
                <span
                  className={`tag ${item.verdict === 'authentic' ? 'success' : item.verdict === 'manipulated' ? 'danger' : 'warning'}`}
                >
                  {item.verdict === 'authentic'
                    ? 'NOT FLAGGED'
                    : item.verdict.toUpperCase()}
                </span>
                <button
                  className="icon-button"
                  aria-label={`Delete ${item.filename}`}
                  onClick={() => deleteHistoryItem(item.id)}
                >
                  <Trash2 />
                </button>
              </div>
            </article>
          );
        })
      )}
    </div>
  );
}
