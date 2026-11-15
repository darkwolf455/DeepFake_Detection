import { useEffect, useRef } from 'react';
import { Download, FileText, Printer, X } from 'lucide-react';
import { AnalyzeResponse } from '../types';

export function ReportModal({
  result,
  filename,
  onClose,
}: {
  result: AnalyzeResponse;
  filename: string;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify({ ...result, filename }, null, 2)], {
        type: 'application/json',
      }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `verisense_${result.id}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <dialog
      ref={dialog}
      className="report-dialog"
      aria-labelledby="report-title"
      onCancel={onClose}
    >
      <div className="panel-heading">
        <h2 id="report-title" className="section-title">
          <FileText /> Analysis report
        </h2>
        <button
          className="icon-button"
          aria-label="Close report"
          onClick={onClose}
        >
          <X />
        </button>
      </div>
      <div id="printable-certificate" className="stack">
        <div>
          <div className="eyebrow">VERISENSE · STUDENT RESEARCH</div>
          <h3 style={{ fontSize: 24, marginTop: 8 }}>{filename}</h3>
          <p className="small">
            Verdict: {result.verdict} · Decision score:{' '}
            {Math.round(result.confidence * 100)}% · {result.processing_ms} ms
          </p>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Branch</th>
                <th>Manipulation score</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.scores)
                .filter(([, score]) => score !== undefined)
                .map(([name, score]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{Math.round(score! * 100)}%</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3 className="small">Models used</h3>
          <ul
            className="small"
            style={{ color: 'var(--text-secondary)', paddingLeft: 17 }}
          >
            {result.models.map((model, index) => (
              <li key={index}>
                {model.name} · {model.version}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="small">Analysis limitations</h3>
          <ul
            className="small"
            style={{ color: 'var(--text-secondary)', paddingLeft: 17 }}
          >
            {(result.warnings ?? []).map((warning, index) => (
              <li key={index} style={{ marginTop: 8 }}>
                {warning}
              </li>
            ))}
          </ul>
        </div>
        <pre>
          Report: {result.id}
          {'\n'}SHA-256: {result.file_sha256 ?? 'Not reported'}
        </pre>
        <p className="small">
          Model scores are uncalibrated. This report does not certify
          authenticity or prove fabrication.
        </p>
      </div>
      <div
        className="actions"
        style={{
          borderTop: '1px solid var(--border)',
          marginTop: 24,
          paddingTop: 18,
        }}
      >
        <button className="primary-button" onClick={download}>
          <Download size={14} />
          Export JSON
        </button>
        <button className="secondary-button" onClick={() => window.print()}>
          <Printer size={14} />
          Print / save PDF
        </button>
      </div>
    </dialog>
  );
}
