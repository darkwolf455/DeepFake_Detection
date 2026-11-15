import assert from 'node:assert/strict';
import { afterEach, mock, test } from 'node:test';
import { analyzeMediaFile, ApiError, checkSystemHealth, checkSystemModels } from '../src/services/api';

// Representative FastAPI payload, including Pydantic's null optional fields.
const response = {
  id: 'test-image', modality: 'image', verdict: 'inconclusive', confidence: 0.5,
  scores: { image: 0.5, video: null, audio: null, av_sync: null },
  timeline: null, heatmap_url: null, warnings: null, file_sha256: null,
  models: [{ name: 'Test detector', version: 'test', modality: 'image' }],
  processing_ms: 1,
};
const file = new File(['test fixture'], 'image.jpg', { type: 'image/jpeg' });
const reply = (body: unknown, status = 200) => {
  const fetchMock = mock.method(globalThis, 'fetch', async (_input: string | URL | Request, _init?: RequestInit) =>
    new Response(JSON.stringify(body), { status }));
  return fetchMock;
};

afterEach(() => mock.restoreAll());

test('reads live model availability and normalizes an unavailable cutoff', async () => {
  const fetchMock = reply({ device: 'cpu', models: [
    { name: 'LAV-DF Five-Score Fusion', type: 'fusion', status: 'active', decision_threshold: .5 },
    { name: 'Speech classifier', type: 'audio', status: 'unavailable', decision_threshold: null },
  ] });
  const result = await checkSystemModels('http://127.0.0.1:8000/');
  assert.equal(fetchMock.mock.calls[0].arguments[0], 'http://127.0.0.1:8000/api/models');
  assert.equal(result.models[0].decision_threshold, .5);
  assert.equal(result.models[1].decision_threshold, undefined);
});

test('rejects a malformed registry instead of displaying invented availability', async () => {
  reply({ device: 'cpu', models: [{ name: 'Broken', type: 'fusion', status: 'active', decision_threshold: 4 }] });
  await assert.rejects(checkSystemModels(), (error: ApiError) => error.errorCode === 'ERR_MODELS');
});

test('accepts Python null fields and sends multipart data to the selected API', async () => {
  const fetchMock = reply(response);
  const result = await analyzeMediaFile(file, 'http://127.0.0.1:8000/');
  assert.equal(result.scores.video, undefined);
  assert.equal(result.timeline, undefined);
  assert.equal(result.heatmap_url, undefined);
  assert.equal(result.warnings, undefined);
  const [url, options] = fetchMock.mock.calls[0].arguments as unknown as [string, RequestInit];
  assert.equal(url, 'http://127.0.0.1:8000/api/analyze');
  assert.equal(options.method, 'POST');
  assert.equal((options.body as FormData).get('file') instanceof File, true);
});

test('uses the same-origin API proxy by default', async () => {
  const fetchMock = reply(response);
  await analyzeMediaFile(file);
  assert.equal(fetchMock.mock.calls[0].arguments[0], '/api/analyze');
});

test('requests combined analysis explicitly and preserves the fusion score', async () => {
  const fetchMock = reply({ ...response, modality: 'video', scores: { image: .2, video: .3, audio: .8, fusion: .7 } });
  const result = await analyzeMediaFile(file, '', undefined, true);
  assert.equal(fetchMock.mock.calls[0].arguments[0], '/api/analyze?use_fusion=true');
  assert.equal(result.scores.fusion, .7);
  assert.equal(result.scores.av_sync, undefined);
});

test('resolves heatmaps against the API origin', async () => {
  reply({ ...response, heatmap_url: '/heatmaps/test-image.png' });
  const result = await analyzeMediaFile(file, 'https://api.example.test');
  assert.equal(result.heatmap_url, 'https://api.example.test/heatmaps/test-image.png');
});

test('keeps same-origin heatmap links relative when explicitly configured', async () => {
  reply({ ...response, heatmap_url: '/heatmaps/test-image.png' });
  const result = await analyzeMediaFile(file, '');
  assert.equal(result.heatmap_url, '/heatmaps/test-image.png');
});

test('rejects invalid probabilities rather than displaying a verdict', async () => {
  reply({ ...response, scores: { image: 1.5 } });
  await assert.rejects(analyzeMediaFile(file), (error: ApiError) =>
    error.errorCode === 'ERR_SCHEMA_VALIDATION');
});

test('preserves unavailable-model errors from the backend', async () => {
  reply({ detail: 'Image checkpoint unavailable.', error_code: 'ERR_503' }, 503);
  await assert.rejects(analyzeMediaFile(file), (error: ApiError) =>
    error.message === 'Image checkpoint unavailable.' && error.errorCode === 'ERR_503');
});

test('handles FastAPI validation arrays with a readable error', async () => {
  reply({ detail: [{ msg: 'Field required', loc: ['body', 'file'] }] }, 422);
  await assert.rejects(analyzeMediaFile(file), (error: ApiError) =>
    error.errorCode === 'ERR_422' && !error.message.includes('[object Object]'));
});

test('reports cancellation without a network failure', async () => {
  mock.method(globalThis, 'fetch', async () => {
    throw new DOMException('Aborted', 'AbortError');
  });
  await assert.rejects(analyzeMediaFile(file), (error: ApiError) =>
    error.errorCode === 'ERR_ABORTED');
});

test('reports offline health when the API cannot be reached', async () => {
  mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  const health = await checkSystemHealth();
  assert.equal(health.status, 'offline');
  assert.deepEqual(health.active_branches, []);
});

test('rejects malformed health data instead of breaking the status display', async () => {
  reply({ status: 'ok' });
  const health = await checkSystemHealth();
  assert.equal(health.status, 'offline');
});

test('displays backend-reported health and applies a request timeout', async () => {
  const data = { status: 'ok', device: 'cpu', active_branches: [], version: '1.0.0' };
  const fetchMock = reply(data);
  assert.deepEqual(await checkSystemHealth(), data);
  const [, options] = fetchMock.mock.calls[0].arguments as unknown as [string, RequestInit];
  assert.equal(options.signal instanceof AbortSignal, true);
});
