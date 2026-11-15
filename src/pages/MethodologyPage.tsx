import {
  AudioLines,
  BookOpen,
  Fingerprint,
  Layers3,
  ScanLine,
} from 'lucide-react';

const steps = [
  {
    name: 'Inspect the media',
    icon: Fingerprint,
    detail:
      'Bounded uploads, file fingerprinting and temporary storage with cleanup after processing.',
  },
  {
    name: 'Extract the signals',
    icon: ScanLine,
    detail:
      'CLIP face features, sampled temporal sequences and frozen Wav2Vec2 speech features.',
  },
  {
    name: 'Combine evidence',
    icon: Layers3,
    detail:
      'Optional video/audio fusion uses image mean/max, a temporal score and audio mean/max.',
  },
  {
    name: 'Read the decision',
    icon: AudioLines,
    detail:
      'Validation-selected operating cutoffs, real branch scores and explicit limitations. No lip-sync analysis.',
  },
];
const benchmarks = [
  ['FaceForensics++ Deepfakes', 'Image', '200 face crops', '90.00%', '0.9714'],
  ['FaceForensics++ Deepfakes', 'Video', '32 videos', '84.38%', '0.9688'],
  ['ASVspoof 2019 LA', 'Audio', '200 clips', '97.00%', '0.9986'],
  ['LAV-DF subset', 'Fusion', '92 scored / 100 clips', '78.26%', '0.9074'],
];

export function MethodologyPage() {
  return (
    <div className="page stack">
      <section className="page-heading">
        <div className="eyebrow">
          <BookOpen size={13} /> Understand what is measured
        </div>
        <h1>Behind every signal.</h1>
        <p>
          A small, reproducible research project. Frozen foundation models,
          compact trained heads, and clearly bounded claims.
        </p>
      </section>
      <div className="score-grid">
        {steps.map(({ name, icon: Icon, detail }, index) => (
          <section className="panel method-step" key={name}>
            <span className="step-number">
              0{index + 1}{' '}
              <Icon size={16} style={{ display: 'inline', marginLeft: 8 }} />
            </span>
            <h3>{name}</h3>
            <p>{detail}</p>
          </section>
        ))}
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2 className="section-title">
              <Layers3 /> Recorded test results
            </h2>
            <p>
              Small benchmark subsets. Accuracy here does not establish
              reliability on arbitrary uploads.
            </p>
          </div>
          <span className="tag">MEASURED SCOPE</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Branch</th>
                <th>Test size</th>
                <th>Accuracy</th>
                <th>AUC</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map(([dataset, branch, size, accuracy, auc]) => (
                <tr key={branch}>
                  <td>{dataset}</td>
                  <td>{branch}</td>
                  <td>{size}</td>
                  <td>{accuracy}</td>
                  <td>{auc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="two-column">
        <section className="panel">
          <div className="eyebrow">Experimental fusion</div>
          <h2 style={{ fontSize: 22, margin: '13px 0' }}>
            Better combined. Still imperfect.
          </h2>
          <p className="small">
            LAV-DF visual and audio heads were adapted on training clips. Fusion
            learned from predictions made with source families held out of each
            fold. The cutoff of 0.5 was selected on validation.
          </p>
          <div className="result-meta">
            <span>81.16% BALANCED ACCURACY</span>
            <span>8 / 100 REJECTED</span>
          </div>
          <p
            className="small"
            style={{ marginTop: 16, color: 'var(--warning)' }}
          >
            Video-only manipulation accuracy was 43.48%. Audio-only and combined
            manipulation accuracy were each 91.30% on usable clips.
          </p>
        </section>
        <section className="panel">
          <div className="eyebrow" style={{ color: 'var(--violet)' }}>
            Know the boundaries
          </div>
          <h2 style={{ fontSize: 22, margin: '13px 0' }}>
            A score is not a guarantee.
          </h2>
          <ul
            className="small"
            style={{
              paddingLeft: 17,
              color: 'var(--text-secondary)',
              display: 'grid',
              gap: 12,
            }}
          >
            <li>
              The image model was trained on face manipulation, not broad modern
              AI-image generation. It has accepted an AI portrait during manual
              testing.
            </li>
            <li>
              Related crops, windows and manipulated variants are correlated.
              Source families are separated; LAV-DF speaker separation is
              unverified.
            </li>
            <li>
              The LAV-DF test cohort had already been inspected for the initial
              fusion baseline. Rejections are abstentions, not correct
              predictions.
            </li>
            <li>
              Scores are uncalibrated. Heatmaps and lip-sync are not
              implemented.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
