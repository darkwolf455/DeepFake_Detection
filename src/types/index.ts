import { z } from 'zod';

export const ModalitySchema = z.enum(['image', 'video', 'audio']);
export type Modality = z.infer<typeof ModalitySchema>;

export const VerdictSchema = z.enum(['authentic', 'manipulated', 'inconclusive']);
export type Verdict = z.infer<typeof VerdictSchema>;

export const TimelinePointSchema = z.object({
  t: z.number(),
  fake_prob: z.number().min(0).max(1),
});
export type TimelinePoint = z.infer<typeof TimelinePointSchema>;

export const ModelVersionSchema = z.object({
  name: z.string(),
  version: z.string(),
  modality: z.string().optional(),
  status: z.string().optional(),
});
export type ModelVersion = z.infer<typeof ModelVersionSchema>;

export const ModalityScoresSchema = z.object({
  image: z.number().min(0).max(1).optional(),
  video: z.number().min(0).max(1).optional(),
  audio: z.number().min(0).max(1).optional(),
  av_sync: z.number().min(0).max(1).optional(),
});
export type ModalityScores = z.infer<typeof ModalityScoresSchema>;

export const AnalyzeResponseSchema = z.object({
  id: z.string(),
  modality: ModalitySchema,
  verdict: VerdictSchema,
  confidence: z.number().min(0).max(1),
  scores: ModalityScoresSchema,
  timeline: z.array(TimelinePointSchema).optional(),
  heatmap_url: z.string().optional(),
  models: z.array(ModelVersionSchema),
  processing_ms: z.number(),
  warnings: z.array(z.string()).optional(),
  file_sha256: z.string().optional(),
});
export type AnalyzeResponse = z.infer<typeof AnalyzeResponseSchema>;

export interface HistoryItem {
  id: string;
  filename: string;
  modality: Modality;
  verdict: Verdict;
  confidence: number;
  timestamp: string;
  scores: ModalityScores;
  file_sha256?: string;
}

export interface AppSettings {
  apiBaseUrl: string;
  theme: 'dark' | 'light';
  inconclusiveLower: number;
  inconclusiveUpper: number;
  enableHeatmapOverlay: boolean;
}
