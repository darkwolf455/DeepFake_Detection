import { useEffect, useRef, useState } from 'react';
import {
  ArrowUpRight,
  AudioLines,
  Check,
  CircleAlert,
  Cpu,
  FileImage,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  ScanLine,
  Upload,
  Video,
  X,
} from 'lucide-react';
import { analyzeMediaFile, checkSystemHealth } from '../services/api';
import { AnalyzeResponse, Modality } from '../types';
import { MediaPreview } from '../components/MediaPreview';
import { ScannerScene } from '../components/ScannerScene';

export function AnalyzePage({
  apiBaseUrl,
  health,
  onAnalysisComplete,
}: {
  apiBaseUrl: string;
  health: Awaited<ReturnType<typeof checkSystemHealth>> | null;
  onAnalysisComplete: (result: AnalyzeResponse, file: File) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState<Modality>('image');
  const [preview, setPreview] = useState('');
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [fusion, setFusion] = useState(true);
  const input = useRef<HTMLInputElement>(null);
  const request = useRef<AbortController | null>(null);
  const active = health?.active_branches ?? [];
  const combined = modality === 'video' && active.includes('fusion') && fusion;
  useEffect(() => {
    const url = file ? URL.createObjectURL(file) : '';
    setPreview(url);
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [file]);
  useEffect(
    () => () => {
      request.current?.abort();
    },
    [],
  );

  const choose = (selected?: File) => {
    if (!selected) return;
    const extension = selected.name.split('.').pop()?.toLowerCase() ?? '';
    const type: Modality | null = [
      'jpg',
      'jpeg',
      'png',
      'webp',
      'bmp',
    ].includes(extension)
      ? 'image'
      : ['mp4', 'mov', 'avi', 'webm', 'mkv'].includes(extension)
        ? 'video'
        : ['wav', 'flac'].includes(extension)
          ? 'audio'
          : null;
    if (!type) {
      setError(
        'Choose an image, video, or WAV/FLAC speech clip from the supported formats.',
      );
      return;
    }
    if (!selected.size || selected.size > 25 * 1024 * 1024) {
      setError('Choose a non-empty file no larger than 25 MB.');
      return;
    }
    setError('');
    setFile(selected);
    setModality(type);
  };
  const analyze = async () => {
    if (!file || busy) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError('');
    try {
      const result = await analyzeMediaFile(
        file,
        apiBaseUrl,
        controller.signal,
        combined,
      );
      if (!controller.signal.aborted) onAnalysisComplete(result, file);
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          failure instanceof Error
            ? failure.message
            : 'Analysis could not complete.',
        );
    } finally {
      if (request.current === controller) {
        setBusy(false);
        request.current = null;
      }
    }
  };
  const modules = [
    { id: 'image', name: 'Face image', icon: FileImage },
    { id: 'video', name: 'Temporal video', icon: Video },
    { id: 'audio', name: 'Speech audio', icon: AudioLines },
    { id: 'fusion', name: 'Video + audio fusion', icon: Layers3 },
  ];

  return (
    <div className="page">
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <ScanLine size={13} /> A closer look at digital media
          </div>
          <h1>
            More than meets
            <br />
            <span>the eye.</span>
          </h1>
          <p>
            Inspect faces, trace speech patterns, and bring video and audio
            evidence together in one private research workspace.
          </p>
          <div className="hero-tags">
            <span className="tag">IMAGE</span>
            <span className="tag">VIDEO</span>
            <span className="tag">AUDIO</span>
            <span className="tag">EXPERIMENTAL FUSION</span>
          </div>
        </div>
        <ScannerScene />
      </section>
      {error && (
        <div role="alert" className="alert error" style={{ marginBottom: 20 }}>
          <CircleAlert />
          <span style={{ flex: 1 }}>{error}</span>
          <button
            className="icon-button"
            aria-label="Dismiss error"
            onClick={() => setError('')}
          >
            <X />
          </button>
        </div>
      )}
      <div className="analysis-grid">
        <section className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <h2 className="section-title">
                <Upload />{' '}
                {file ? 'Ready for a closer look' : 'Start an analysis'}
              </h2>
              <p>
                {file
                  ? 'Preview your media and choose how to inspect it.'
                  : 'Your file is processed locally by the Python engine.'}
              </p>
            </div>
            <span className="tag">01 / UPLOAD</span>
          </div>
          <input
            ref={input}
            type="file"
            className="sr-only"
            aria-label="Choose media file"
            accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.avi,.webm,.mkv,.wav,.flac"
            onChange={(event) => {
              choose(event.target.files?.[0]);
              event.target.value = '';
            }}
          />
          {!file ? (
            <>
              <button
                className={`upload-zone ${dragging ? 'dragging' : ''}`}
                onClick={() => input.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  if (event.dataTransfer.files.length !== 1)
                    setError('Drop one file at a time.');
                  else choose(event.dataTransfer.files[0]);
                }}
              >
                <span className="upload-symbol">
                  <Upload strokeWidth={1.5} />
                </span>
                <strong>Drop your media here</strong>
                <span>or browse a file from your device</span>
                <span className="secondary-button" style={{ marginTop: 9 }}>
                  Choose a file <ArrowUpRight size={14} />
                </span>
              </button>
              <div className="file-types">
                <span>JPG · PNG · WEBP</span>
                <span>MP4 · MOV · WEBM</span>
                <span>WAV · FLAC</span>
              </div>
            </>
          ) : (
            <div className="stack" style={{ gap: 18 }}>
              <div className="panel-heading" style={{ marginBottom: 0 }}>
                <div className="file-heading">
                  <Check />
                  <div>
                    <strong>{file.name}</strong>
                    <span>
                      {(file.size / 1024 / 1024).toFixed(2)} MB ·{' '}
                      {modality.toUpperCase()}
                    </span>
                  </div>
                </div>
                <button
                  className="icon-button"
                  aria-label="Remove selected file"
                  disabled={busy}
                  onClick={() => {
                    setFile(null);
                    setError('');
                  }}
                >
                  <X />
                </button>
              </div>
              {preview && (
                <MediaPreview
                  src={preview}
                  modality={modality}
                  filename={file.name}
                />
              )}
              {modality === 'video' && active.includes('fusion') && (
                <label className="fusion-option" style={{ margin: 0 }}>
                  <input
                    type="checkbox"
                    checked={fusion}
                    disabled={busy}
                    onChange={(event) => setFusion(event.target.checked)}
                  />
                  <span>
                    Combine video and audio{' '}
                    <span className="tag">EXPERIMENTAL</span>
                    <small>
                      Requires a clear face and audible speech, up to 30
                      seconds. Turn off for visual-only scoring.
                    </small>
                  </span>
                </label>
              )}
              {busy ? (
                <div className="processing" role="status">
                  <LoaderCircle className="spinner" size={19} />
                  <span>
                    Examining your media…
                    <small style={{ display: 'block', color: 'var(--muted)' }}>
                      CPU analysis may take a moment.
                    </small>
                  </span>
                  <button
                    className="secondary-button"
                    onClick={() => request.current?.abort()}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  className="primary-button"
                  onClick={analyze}
                  disabled={!active.includes(combined ? 'fusion' : modality)}
                >
                  <ScanLine size={17} />{' '}
                  {combined ? 'Analyze video + audio' : `Analyze ${modality}`}{' '}
                  <ArrowUpRight size={15} />
                </button>
              )}
              {!active.includes(combined ? 'fusion' : modality) && (
                <p className="small" role="status">
                  This model is unavailable. Check the engine connection in
                  Settings.
                </p>
              )}
            </div>
          )}
          <div className="upload-note">
            <LockKeyhole /> Temporary uploads are removed after processing.
            Results stay in this browser.
          </div>
        </section>
        <aside className="stack" style={{ alignContent: 'start' }}>
          <section className="panel">
            <div className="panel-heading">
              <h2 className="section-title">
                <Cpu /> Detection engine
              </h2>
              <span className={`tag ${active.length ? 'success' : ''}`}>
                <span className="status-dot" />
                {!health ? 'Checking' : active.length ? 'Online' : 'Offline'}
              </span>
            </div>
            <dl className="engine-summary">
              <div>
                <dt>COMPUTE</dt>
                <dd>{health?.device ?? '—'}</dd>
              </div>
              <div>
                <dt>FILE LIMIT</dt>
                <dd>25 MB</dd>
              </div>
            </dl>
            <div className="engine-list">
              {modules.map(({ id, name, icon: Icon }) => (
                <div className="engine-row" key={id}>
                  <span>
                    <Icon />
                    {name}
                  </span>
                  <span
                    className={`engine-label ${active.includes(id) ? 'active' : ''}`}
                  >
                    {active.includes(id) ? 'LOADED' : 'UNAVAILABLE'}
                  </span>
                </div>
              ))}
            </div>
          </section>
          <section
            className="panel"
            style={{
              background:
                'linear-gradient(135deg, var(--violet-soft), transparent), var(--surface)',
            }}
          >
            <div className="eyebrow" style={{ color: 'var(--violet)' }}>
              A research tool, with limits
            </div>
            <h3 style={{ fontSize: 18, fontWeight: 550, margin: '12px 0 9px' }}>
              Signals, not certainty.
            </h3>
            <p className="small">
              Scores reflect small training datasets. Unfamiliar generators and
              compressed media can fool the models.
            </p>
          </section>
        </aside>
      </div>
    </div>
  );
}
