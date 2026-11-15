import { useState } from 'react';
import { AudioLines } from 'lucide-react';
import { Modality } from '../types';

export function MediaPreview({
  src,
  modality,
  filename,
}: {
  src: string;
  modality: Modality;
  filename: string;
}) {
  const [failedSource, setFailedSource] = useState<string | null>(null);
  const failed = () => setFailedSource(src);
  return (
    <div className="media-preview">
      {modality === 'image' ? (
        <img src={src} alt={`Preview of ${filename}`} onError={failed} />
      ) : modality === 'video' ? (
        <video
          key={src}
          src={src}
          controls
          playsInline
          preload="metadata"
          onError={failed}
          aria-label={`Preview of ${filename}`}
        />
      ) : (
        <>
          <AudioLines />
          <audio
            key={src}
            src={src}
            controls
            preload="metadata"
            onError={failed}
            aria-label={`Listen to ${filename}`}
          />
        </>
      )}
      {failedSource === src && (
        <p role="status" className="small" style={{ color: 'var(--warning)' }}>
          This browser cannot preview the file. For video playback, try an H.264
          MP4. The detector may still be able to analyze it.
        </p>
      )}
    </div>
  );
}
