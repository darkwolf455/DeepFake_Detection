import { useEffect, useState } from 'react';
import {
  AudioLines,
  Boxes,
  CircleAlert,
  Cpu,
  FileImage,
  Layers3,
  RefreshCw,
  Video,
} from 'lucide-react';
import { checkSystemModels } from '../services/api';

const descriptions: Record<string, string> = {
  image:
    'Frozen CLIP ViT-L/14 with a small face classifier. One detected face per image.',
  video:
    'Sampled face features and a temporal classifier. Visual-only analysis keeps the soundtrack separate.',
  audio:
    'Frozen Wav2Vec2 Base with learned pooling and a speech classifier. Central four seconds.',
  fusion:
    'Adapted visual/audio heads and five-score logistic fusion. Requires both usable faces and a soundtrack.',
};
const icons: Record<string, typeof Cpu> = {
  image: FileImage,
  video: Video,
  audio: AudioLines,
  fusion: Layers3,
};

export function ModelsPage({ apiBaseUrl }: { apiBaseUrl: string }) {
  const [data, setData] = useState<Awaited<
    ReturnType<typeof checkSystemModels>
  > | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const refresh = async () => {
    setLoading(true);
    setError('');
    try {
      setData(await checkSystemModels(apiBaseUrl));
    } catch (failure) {
      setData(null);
      setError(
        failure instanceof Error
          ? failure.message
          : 'Could not load the model registry.',
      );
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    refresh();
  }, [apiBaseUrl]);
  return (
    <div className="page stack">
      <section className="page-heading">
        <div className="eyebrow">
          <Boxes size={13} /> Under the surface
        </div>
        <h1>The models behind the signals.</h1>
        <p>
          Live availability from your Python engine. Each branch has a different
          measured scope.
        </p>
      </section>
      <div className="actions" style={{ justifyContent: 'space-between' }}>
        <span className="tag">
          <Cpu size={12} /> COMPUTE: {data?.device ?? '—'}
        </span>
        <button
          className="secondary-button"
          disabled={loading}
          onClick={refresh}
        >
          <RefreshCw size={14} className={loading ? 'spinner' : ''} />
          Refresh registry
        </button>
      </div>
      {error && (
        <div role="alert" className="alert error">
          <CircleAlert />
          {error}
        </div>
      )}
      {loading && !data && (
        <p role="status" className="small">
          Reading model availability…
        </p>
      )}
      <div className="model-grid">
        {data?.models.map((model) => {
          const Icon = icons[model.type] ?? Cpu;
          return (
            <article className="panel model-card" key={model.type}>
              <div
                className="actions"
                style={{ justifyContent: 'space-between', alignItems: 'start' }}
              >
                <div className="model-icon">
                  <Icon size={23} strokeWidth={1.5} />
                </div>
                <span
                  className={`tag ${model.status === 'active' ? 'success' : ''}`}
                >
                  <span className="status-dot" />
                  {model.status.toUpperCase()}
                </span>
              </div>
              <h2>{model.name}</h2>
              <p>{descriptions[model.type]}</p>
              <div className="model-facts">
                <span>{model.type.toUpperCase()}</span>
                <span>
                  {model.decision_threshold !== undefined
                    ? `CUTOFF ${model.decision_threshold.toFixed(4)}`
                    : 'NO FIXED CUTOFF REPORTED'}
                </span>
              </div>
            </article>
          );
        })}
      </div>
      <p className="small muted">
        Loading a model confirms availability. It does not establish reliability
        on arbitrary or newly generated media.
      </p>
    </div>
  );
}
