import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, RotateCcw } from 'lucide-react';

interface AudioWaveformProps {
  audioUrl: string;
  onTimeUpdate?: (currentTime: number) => void;
  seekTime?: number | null;
}

export const AudioWaveform: React.FC<AudioWaveformProps> = ({
  audioUrl,
  onTimeUpdate,
  seekTime,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [bars] = useState<number[]>(() =>
    Array.from({ length: 64 }, () => 0.15 + Math.random() * 0.75)
  );

  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && audioRef.current) {
      audioRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (!audioRef.current) return;
    const cur = audioRef.current.currentTime;
    setCurrentTime(cur);
    if (onTimeUpdate) onTimeUpdate(cur);
  };

  const handleLoadedMetadata = () => {
    if (!audioRef.current) return;
    setDuration(audioRef.current.duration || 10);
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const target = ratio * duration;
    audioRef.current.currentTime = target;
    setCurrentTime(target);
    if (onTimeUpdate) onTimeUpdate(target);
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const progressRatio = duration > 0 ? currentTime / duration : 0;

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        muted={isMuted}
      />

      {/* Top player controls */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="w-10 h-10 rounded-xl bg-gradient-to-r from-cyan-500 to-violet-600 text-white flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-md shadow-cyan-500/25"
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
          </button>

          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = 0;
                setCurrentTime(0);
              }
            }}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            title="Restart"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <div className="font-mono text-xs text-slate-300">
            <span>{formatTime(currentTime)}</span>
            <span className="text-slate-500 mx-1">/</span>
            <span className="text-slate-500">{formatTime(duration)}</span>
          </div>
        </div>

        <button
          onClick={() => setIsMuted(!isMuted)}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          aria-label={isMuted ? 'Unmute' : 'Mute'}
        >
          {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4" />}
        </button>
      </div>

      {/* Waveform Visualization Bars */}
      <div
        onClick={handleSeek}
        className="relative h-16 w-full flex items-center justify-between gap-[2px] cursor-pointer group bg-slate-950/60 rounded-xl px-3 py-2 border border-slate-800/80 overflow-hidden"
      >
        {bars.map((heightFactor, idx) => {
          const barRatio = idx / bars.length;
          const isPassed = barRatio <= progressRatio;
          return (
            <div
              key={idx}
              className={`w-full rounded-full transition-all duration-75 ${
                isPassed
                  ? 'bg-gradient-to-t from-cyan-400 to-violet-400 shadow-sm shadow-cyan-500/50'
                  : 'bg-slate-700/60 group-hover:bg-slate-600/70'
              }`}
              style={{
                height: `${Math.max(12, heightFactor * 100)}%`,
              }}
            />
          );
        })}

        {/* Current Seek Head Line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 pointer-events-none shadow-sm shadow-cyan-400"
          style={{ left: `${progressRatio * 100}%` }}
        />
      </div>

      <div className="flex justify-between items-center mt-2 text-[11px] text-slate-500 font-mono">
        <span>16kHz Mono • Spectral Density</span>
        <span>Click waveform to seek</span>
      </div>
    </div>
  );
};
