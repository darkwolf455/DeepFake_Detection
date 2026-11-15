import { z } from 'zod';

// FastAPI serializes absent optional fields as null; the UI uses undefined.
const optionalField = <T extends z.ZodType>(schema: T) =>
  schema.nullish().transform((value) => value ?? undefined);

export const ModalitySchema = z.enum(['image', 'video', 'audio']);
export type Modality = z.infer<typeof ModalitySchema>;

export const VerdictSchema = z.enum([
  'authentic',
  'manipulated',
  'inconclusive',
]);
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
  image: optionalField(z.number().min(0).max(1)),
  video: optionalField(z.number().min(0).max(1)),
  audio: optionalField(z.number().min(0).max(1)),
  av_sync: optionalField(z.number().min(0).max(1)),
  fusion: optionalField(z.number().min(0).max(1)),
});
export type ModalityScores = z.infer<typeof ModalityScoresSchema>;

export const AnalyzeResponseSchema = z.object({
  id: z.string(),
  modality: ModalitySchema,
  verdict: VerdictSchema,
  confidence: z.number().min(0).max(1),
  scores: ModalityScoresSchema,
  timeline: optionalField(z.array(TimelinePointSchema)),
  heatmap_url: optionalField(z.string()),
  models: z.array(ModelVersionSchema),
  processing_ms: z.number(),
  warnings: optionalField(z.array(z.string())),
  file_sha256: optionalField(z.string()),
});
export type AnalyzeResponse = z.infer<typeof AnalyzeResponseSchema>;

export const HealthResponseSchema = z.object({
  status: z.string(),
  device: z.string(),
  active_branches: z.array(z.string()),
  version: z.string(),
});

export const ModelsResponseSchema = z.object({
  device: z.string(),
  models: z.array(
    z.object({
      name: z.string(),
      type: z.string(),
      status: z.enum(['active', 'unavailable']),
      decision_threshold: optionalField(z.number().min(0).max(1)),
    }),
  ),
});

export const HistoryItemSchema = z.object({
  id: z.string(),
  filename: z.string(),
  modality: ModalitySchema,
  verdict: VerdictSchema,
  confidence: z.number().min(0).max(1),
  timestamp: z.string(),
  scores: ModalityScoresSchema,
  file_sha256: optionalField(z.string()),
});
export type HistoryItem = z.infer<typeof HistoryItemSchema>;
