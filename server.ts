import express from 'express';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = Number(process.env.PORT || 3000);
const backend = new URL(process.env.BACKEND_URL || 'http://127.0.0.1:8000');
if (backend.protocol !== 'http:')
  throw new Error('BACKEND_URL must use http://');

// Stream uploads to Python without buffering or inventing inference results.
app.use(['/api', '/health'], (req, res) => {
  const upstream = http.request(
    new URL(req.originalUrl, backend),
    {
      method: req.method,
      headers: { ...req.headers, host: backend.host },
    },
    (response) => {
      res.writeHead(response.statusCode || 502, response.headers);
      response.on('error', () => res.destroy());
      response.pipe(res);
    },
  );
  upstream.setTimeout(120_000, () =>
    upstream.destroy(new Error('Backend timeout')),
  );
  upstream.on('error', () => {
    if (!res.headersSent) {
      res
        .status(502)
        .json({
          detail: 'Python backend unavailable. Start the API on port 8000.',
          error_code: 'ERR_BACKEND_UNAVAILABLE',
        });
    } else res.destroy();
  });
  req.on('aborted', () => upstream.destroy());
  res.on('close', () => {
    if (!res.writableEnded) upstream.destroy();
  });
  req.pipe(upstream);
});

if (process.env.NODE_ENV !== 'production') {
  const { createServer } = await import('vite');
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
} else {
  app.use(express.static(path.join(directory, 'dist')));
  app.get('*', (_req, res) =>
    res.sendFile(path.join(directory, 'dist/index.html')),
  );
}

app.listen(port, '127.0.0.1', () => {
  console.log(
    `[VeriSense] Open http://localhost:${port}; Python API: ${backend.origin}`,
  );
});
