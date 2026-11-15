import {
  AnalyzeResponse,
  AnalyzeResponseSchema,
  HealthResponseSchema,
  ModelsResponseSchema,
} from '../types';

export const DEFAULT_API_BASE_URL = import.meta.env?.VITE_API_BASE_URL ?? '';

export class ApiError extends Error {
  errorCode: string;
  constructor(message: string, errorCode: string = 'ERR_UNKNOWN') {
    super(message);
    this.name = 'ApiError';
    this.errorCode = errorCode;
  }
}

export async function checkSystemModels(
  apiBaseUrl: string = DEFAULT_API_BASE_URL,
) {
  const base = apiBaseUrl.trim().replace(/\/$/, '');
  try {
    const response = await fetch(`${base}/api/models`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok)
      throw new Error(
        'The model registry is unavailable. Check your API connection.',
      );
    const parsed = ModelsResponseSchema.safeParse(await response.json());
    if (!parsed.success)
      throw new Error('The engine returned an invalid model registry.');
    return parsed.data;
  } catch (error) {
    throw new ApiError(
      error instanceof Error
        ? error.message
        : 'Could not reach the model registry.',
      'ERR_MODELS',
    );
  }
}

export async function analyzeMediaFile(
  file: File,
  apiBaseUrl: string = DEFAULT_API_BASE_URL,
  signal?: AbortSignal,
  useFusion: boolean = false,
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const cleanBase = apiBaseUrl.trim().replace(/\/$/, '');
  const url =
    (cleanBase ? `${cleanBase}/api/analyze` : '/api/analyze') +
    (useFusion ? '?use_fusion=true' : '');

  try {
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal,
    });

    if (!res.ok) {
      let errorDetail = `Server error (${res.status} ${res.statusText})`;
      let errorCode = `ERR_${res.status}`;
      try {
        const errorJson = await res.json();
        if (typeof errorJson.detail === 'string')
          errorDetail = errorJson.detail;
        if (Array.isArray(errorJson.detail))
          errorDetail =
            'The server rejected the upload. Check the file and try again.';
        if (typeof errorJson.error_code === 'string')
          errorCode = errorJson.error_code;
      } catch {
        // Fallback to text status
      }
      throw new ApiError(errorDetail, errorCode);
    }

    const rawData = await res.json();

    // Validate using Zod schema
    const parseResult = AnalyzeResponseSchema.safeParse(rawData);
    if (!parseResult.success) {
      console.error('API response schema validation error:', parseResult.error);
      throw new ApiError(
        'Server returned an invalid or malformed data payload.',
        'ERR_SCHEMA_VALIDATION',
      );
    }

    const result = parseResult.data;
    if (result.heatmap_url && cleanBase) {
      result.heatmap_url = new URL(result.heatmap_url, `${cleanBase}/`).href;
    }
    return result;
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new ApiError(
        'Analysis request was aborted by user.',
        'ERR_ABORTED',
      );
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      err.message || 'Network failure connecting to forensic analysis server.',
      'ERR_NETWORK',
    );
  }
}

export async function checkSystemHealth(
  apiBaseUrl: string = DEFAULT_API_BASE_URL,
): Promise<{
  status: string;
  device: string;
  active_branches: string[];
  version: string;
}> {
  const cleanBase = apiBaseUrl.trim().replace(/\/$/, '');
  const url = cleanBase ? `${cleanBase}/health` : '/health';

  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error('Health endpoint returned error');
    return HealthResponseSchema.parse(await res.json());
  } catch (err) {
    return {
      status: 'offline',
      device: 'Unknown',
      active_branches: [],
      version: 'unknown',
    };
  }
}
