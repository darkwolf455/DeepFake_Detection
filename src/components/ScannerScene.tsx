import { useRef } from 'react';
import { AudioLines, Fingerprint, Layers3, ScanFace } from 'lucide-react';

/** Decorative CSS geometry: no WebGL context or extra model downloads. */
export function ScannerScene() {
  const scene = useRef<HTMLDivElement>(null);
  return (
    <div
      ref={scene}
      className="scanner-scene"
      aria-hidden="true"
      onPointerMove={(event) => {
        if (event.pointerType === 'touch' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        event.currentTarget.style.setProperty('--pointer-x', `${((event.clientX - bounds.left) / bounds.width - 0.5) * 16}deg`);
        event.currentTarget.style.setProperty('--pointer-y', `${((event.clientY - bounds.top) / bounds.height - 0.5) * -12}deg`);
      }}
      onPointerLeave={() => {
        scene.current?.style.setProperty('--pointer-x', '0deg');
        scene.current?.style.setProperty('--pointer-y', '0deg');
      }}
    >
      <div className="scanner-floor" />
      <div className="scanner-orbit orbit-one" />
      <div className="scanner-orbit orbit-two" />
      <div className="scanner-platform" />
      <div className="scanner-rig">
        <div className="scanner-core">
          <div className="scanner-layer scanner-back" />
          <div className="scanner-layer scanner-middle" />
          <div className="scanner-layer scanner-front">
            <span className="scanner-corner corner-top" />
            <Fingerprint />
            <span className="scan-beam" />
            <span className="scanner-corner corner-bottom" />
          </div>
        </div>
      </div>
      <div className="scene-chip chip-visual"><ScanFace /> Visual signals</div>
      <div className="scene-chip chip-audio"><AudioLines /> Speech patterns</div>
      <div className="scene-chip chip-fusion"><Layers3 /> Combined evidence</div>
      <span className="scene-caption">MULTIMODAL SIGNAL EXPLORER</span>
    </div>
  );
}
