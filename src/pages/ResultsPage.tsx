import { useState } from 'react';
import {
  ArrowUpRight,
  AudioLines,
  CircleAlert,
  FileImage,
  FileText,
  Fingerprint,
  Layers3,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  Video,
} from 'lucide-react';
import { AnalyzeResponse } from '../types';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { MediaPreview } from '../components/MediaPreview';
import { ReportModal } from '../components/ReportModal';
import { TimelineChart } from '../components/TimelineChart';

export function ResultsPage({
  result,
  originalFile,
  previewUrl,
  onReset,
}: {
  result: AnalyzeResponse;
  originalFile: File;
  previewUrl: string;
  onReset: () => void;
}) {
  const [reportOpen, setReportOpen] = useState(false);
  const fused = result.scores.fusion !== undefined;
  const decision = {
    authentic: {
      title: 'No manipulation flagged.',
      detail:
        'The model score is below its decision cutoff. This does not verify authenticity.',
      tone: 'success',
      icon: ShieldCheck,
    },
    manipulated: {
      title: 'Possible manipulation detected.',
      detail:
        'This student model flagged the media. Review the scores and known limitations alongside other evidence.',
      tone: 'danger',
      icon: CircleAlert,
    },
    inconclusive: {
      title: 'The signal is inconclusive.',
      detail:
        'The score falls inside the configured uncertainty band. No confident classification is available.',
      tone: 'warning',
      icon: CircleAlert,
    },
  }[result.verdict];
  const DecisionIcon = decision.icon;
  const cards = [
    {
      key: 'image',
      label: 'Face image',
      icon: FileImage,
      note: fused ? 'Maximum sampled face score' : 'CLIP face classifier',
    },
    {
      key: 'video',
      label: 'Video',
      icon: Video,
      note: result.models.some((model) => model.name.includes('Temporal'))
        ? 'Temporal sequence score'
        : 'Sampled-frame mean',
    },
    {
      key: 'audio',
      label: 'Speech',
      icon: AudioLines,
      note: fused
        ? 'Maximum speech-window score'
        : 'Central four-second segment',
    },
    {
      key: 'fusion',
      label: 'Combined',
      icon: Layers3,
      note: 'Five-score learned fusion',
    },
  ] as const;
  return (
    <div className="page stack">
      <section className="page-heading" style={{ marginBottom: 0 }}>
        <div className="actions" style={{ justifyContent: 'space-between' }}>
          <div>
            <div className="eyebrow">
              <ScanLine size={13} /> Analysis complete
            </div>
            <h1>Your media, examined.</h1>
            <p>
              {fused
                ? 'Combined video + audio analysis'
                : `${result.modality[0].toUpperCase()}${result.modality.slice(1)} analysis`}{' '}
              · {originalFile.name}
            </p>
          </div>
          <div className="actions">
            <button
              className="secondary-button"
              onClick={() => setReportOpen(true)}
            >
              <FileText size={15} />
              View report
            </button>
            <button className="primary-button" onClick={onReset}>
              <RotateCcw size={14} />
              New analysis
            </button>
          </div>
        </div>
      </section>
      <div className="result-overview">
        <section className="panel">
          <span className={`tag ${decision.tone}`}>
            <DecisionIcon size={13} />
            {result.verdict.toUpperCase()}
          </span>
          <h2 className="result-title">{decision.title}</h2>
          <p className="small" style={{ maxWidth: 540 }}>
            {decision.detail}
          </p>
          <div className="result-meta">
            <span>{(result.processing_ms / 1000).toFixed(2)} SECONDS</span>
            <span>{(originalFile.size / 1024 / 1024).toFixed(2)} MB</span>
            <span title={result.id}>ID {result.id.slice(0, 12)}</span>
          </div>
        </section>
        <section className="panel result-gauge">
          <ConfidenceGauge
            confidence={result.confidence}
            verdict={result.verdict}
          />
        </section>
      </div>
      <section>
        <h2 className="section-title" style={{ marginBottom: 15 }}>
          <Layers3 /> The score breakdown
        </h2>
        <div className="score-grid">
          {cards.map(({ key, label, icon: Icon, note }) => {
            const score = result.scores[key];
            return (
              <div className="panel score-card" key={key}>
                <div className="engine-row">
                  <span>
                    <Icon />
                    {label}
                  </span>
                  {score !== undefined && (
                    <ArrowUpRight size={13} style={{ color: 'var(--muted)' }} />
                  )}
                </div>
                <div className="score-value">
                  {score === undefined ? '—' : `${Math.round(score * 100)}%`}
                </div>
                <p style={{ fontSize: 10, minHeight: 30 }}>
                  {score === undefined ? 'Not used for this analysis' : note}
                </p>
                <div className="score-track">
                  <span
                    style={{
                      width: `${(score ?? 0) * 100}%`,
                      background:
                        key === 'fusion' ? 'var(--violet)' : 'var(--accent)',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </section>
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <h2 className="section-title">
              <FileImage /> Original media
            </h2>
            <span className="tag">PREVIEW</span>
          </div>
          <MediaPreview
            src={previewUrl}
            modality={result.modality}
            filename={originalFile.name}
          />
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2 className="section-title">
              <CircleAlert /> Know the limits
            </h2>
          </div>
          <ul
            className="small"
            style={{
              color: 'var(--text-secondary)',
              paddingLeft: 17,
              display: 'grid',
              gap: 13,
            }}
          >
            {(
              result.warnings ?? [
                'Scores are uncalibrated. A decision does not prove authenticity.',
              ]
            ).map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </section>
      </div>
      {result.timeline?.length ? (
        <TimelineChart timeline={result.timeline} />
      ) : null}
      <section className="panel">
        <div className="section-title">
          <Fingerprint /> File fingerprint
        </div>
        <p
          className="font-mono"
          style={{ fontSize: 10, wordBreak: 'break-all', marginTop: 12 }}
        >
          {result.file_sha256 ?? 'No fingerprint reported'}
        </p>
        <p className="small muted" style={{ marginTop: 8 }}>
          SHA-256 identifies this upload; it does not certify its authenticity.
        </p>
      </section>
      {reportOpen && (
        <ReportModal
          result={result}
          filename={originalFile.name}
          onClose={() => setReportOpen(false)}
        />
      )}
    </div>
  );
}
