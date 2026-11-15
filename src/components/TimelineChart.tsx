import { Activity } from 'lucide-react';
import { TimelinePoint } from '../types';

export function TimelineChart({ timeline }: { timeline: TimelinePoint[] }) {
  if (!timeline.length) return null;
  const maxTime = Math.max(1, ...timeline.map((point) => point.t));
  const x = (time: number) => 40 + (time / maxTime) * 720;
  const y = (score: number) => 130 - score * 100;
  const points = timeline
    .map((point) => `${x(point.t)},${y(point.fake_prob)}`)
    .join(' ');
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2 className="section-title">
            <Activity /> Sampled-frame scores
          </h2>
          <p>
            Weak video-level supervision; this is not an edit-localization
            timeline.
          </p>
        </div>
      </div>
      <svg
        viewBox="0 0 800 170"
        role="img"
        aria-label={`${timeline.length} sampled frame scores, first ${maxTime.toFixed(1)} seconds`}
        style={{ width: '100%', maxHeight: 230 }}
      >
        {[0, 0.5, 1].map((value) => (
          <g key={value}>
            <line
              x1="40"
              x2="760"
              y1={y(value)}
              y2={y(value)}
              stroke="var(--border-strong)"
              strokeDasharray="3 5"
            />
            <text
              x="28"
              y={y(value) + 4}
              fill="var(--muted)"
              fontSize="10"
              textAnchor="end"
            >
              {value * 100}%
            </text>
          </g>
        ))}
        <polyline
          points={points}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
        />
        {timeline.map((point, index) => (
          <circle
            key={index}
            cx={x(point.t)}
            cy={y(point.fake_prob)}
            r="3.5"
            fill="var(--accent)"
          >
            <title>
              {point.t.toFixed(2)}s: {(point.fake_prob * 100).toFixed(1)}%
            </title>
          </circle>
        ))}
        <text x="40" y="158" fill="var(--muted)" fontSize="10">
          0s
        </text>
        <text
          x="760"
          y="158"
          fill="var(--muted)"
          fontSize="10"
          textAnchor="end"
        >
          {maxTime.toFixed(1)}s
        </text>
      </svg>
    </section>
  );
}
