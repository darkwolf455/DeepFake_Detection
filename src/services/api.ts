import { AnalyzeResponse, AnalyzeResponseSchema } from '../types';

export class ApiError extends Error {
  errorCode: string;
  constructor(message: string, errorCode: string = 'ERR_UNKNOWN') {
    super(message);
    this.name = 'ApiError';
    this.errorCode = errorCode;
  }
}

export async function analyzeMediaFile(
  file: File,
  apiBaseUrl: string = '',
  signal?: AbortSignal
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const cleanBase = apiBaseUrl.trim().replace(/\/$/, '');
  const url = cleanBase ? `${cleanBase}/api/analyze` : '/api/analyze';

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
        if (errorJson.detail) errorDetail = errorJson.detail;
        if (errorJson.error_code) errorCode = errorJson.error_code;
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
        'ERR_SCHEMA_VALIDATION'
      );
    }

    return parseResult.data;
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new ApiError('Analysis request was aborted by user.', 'ERR_ABORTED');
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      err.message || 'Network failure connecting to forensic analysis server.',
      'ERR_NETWORK'
    );
  }
}

export async function checkSystemHealth(apiBaseUrl: string = ''): Promise<{
  status: string;
  device: string;
  active_branches: string[];
  version: string;
}> {
  const cleanBase = apiBaseUrl.trim().replace(/\/$/, '');
  const url = cleanBase ? `${cleanBase}/health` : '/health';

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Health endpoint returned error');
    return await res.json();
  } catch (err) {
    return {
      status: 'offline',
      device: 'Unknown',
      active_branches: [],
      version: '1.0.0',
    };
  }
}
