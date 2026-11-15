import { Verdict } from '../types';

export function ConfidenceGauge({
  confidence,
  verdict,
}: {
  confidence: number;
  verdict: Verdict;
}) {
  const score = Math.max(0, Math.min(1, confidence));
  const color = `var(--${verdict === 'authentic' ? 'success' : verdict === 'manipulated' ? 'danger' : 'warning'})`;
  const radius = 65,
    length = 2 * Math.PI * radius;
  return (
    <div className="result-gauge">
      <div
        className="gauge"
        role="img"
        aria-label={`Uncalibrated decision score: ${Math.round(score * 100)} percent`}
      >
        <svg viewBox="0 0 150 150" aria-hidden="true">
          <circle
            cx="75"
            cy="75"
            r={radius}
            fill="none"
            stroke="var(--border-strong)"
            strokeWidth="5"
          />
          <circle
            cx="75"
            cy="75"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="5"
            strokeLinecap="round"
            strokeDasharray={length}
            strokeDashoffset={length * (1 - score)}
          />
        </svg>
        <strong style={{ color }}>
          {Math.round(score * 100)}
          <span style={{ fontSize: 13 }}>%</span>
        </strong>
      </div>
      <p className="gauge-caption">Decision score · uncalibrated</p>
    </div>
  );
}
