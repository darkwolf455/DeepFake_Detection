import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { TimelinePoint } from '../types';

interface TimelineChartProps {
  timeline: TimelinePoint[];
  onSeek?: (timestamp: number) => void;
  currentTimestamp?: number;
}

export const TimelineChart: React.FC<TimelineChartProps> = ({
  timeline,
  onSeek,
  currentTimestamp,
}) => {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500 font-mono text-xs">
        No temporal timeline available for this modality.
      </div>
    );
  }

  const chartData = timeline.map((pt) => ({
    time: pt.t,
    fakeProb: Math.round(pt.fake_prob * 100),
    timeLabel: `${pt.t}s`,
  }));

  const handleClick = (e: any) => {
    if (e && e.activePayload && e.activePayload.length > 0 && onSeek) {
      const clickedTime = e.activePayload[0].payload.time;
      onSeek(clickedTime);
    }
  };

  return (
    <div className="w-full bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            Temporal Artifact Timeline
          </h3>
          <p className="text-xs text-slate-400">
            Per-second manipulated probability. Click any point to seek media player.
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            &lt;40% Authentic
          </span>
          <span className="flex items-center gap-1.5 text-amber-400">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            40-65% Uncertain
          </span>
          <span className="flex items-center gap-1.5 text-rose-400">
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            &gt;65% Deepfake
          </span>
        </div>
      </div>

      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            onClick={handleClick}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <defs>
              <linearGradient id="probGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.6} />
                <stop offset="60%" stopColor="#06B6D4" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="timeLabel"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(v) => `${v}%`}
            />

            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-xl shadow-xl text-xs font-mono">
                      <div className="text-slate-400">Timestamp: {data.time}s</div>
                      <div className="text-cyan-400 font-bold mt-1">
                        Manipulated Risk: {data.fakeProb}%
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">Click to seek</div>
                    </div>
                  );
                }
                return null;
              }}
            />

            {/* Threshold lines */}
            <ReferenceLine y={65} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
            <ReferenceLine y={40} stroke="#f59e0b" strokeDasharray="3 3" opacity={0.5} />

            {/* Active seek cursor */}
            {currentTimestamp !== undefined && (
              <ReferenceLine
                x={`${Math.round(currentTimestamp)}s`}
                stroke="#22d3ee"
                strokeWidth={2}
              />
            )}

            <Area
              type="monotone"
              dataKey="fakeProb"
              stroke="#8B5CF6"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#probGradient)"
              activeDot={{ r: 6, fill: '#06B6D4', stroke: '#fff', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
