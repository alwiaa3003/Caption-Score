// Entry point: connect to MongoDB, then start the HTTP server.
import mongoose from 'mongoose';

import { createApp } from './app.js';
import { config } from './config.js';
import { createModelClient } from './services/modelClient.js';

async function main() {
  if (!config.mongodbUri) {
    console.error('MONGODB_URI is not set. Copy .env.example to .env and fill it in.');
    process.exit(1);
  }
  mongoose.connection.on('disconnected', () => console.warn('MongoDB disconnected'));
  mongoose.connection.on('reconnected', () => console.info('MongoDB reconnected'));
  try {
    await mongoose.connect(config.mongodbUri, { serverSelectionTimeoutMS: 10_000 });
  } catch (err) {
    // Never print the URI: it contains the password.
    console.error('Could not connect to MongoDB:', err.message);
    process.exit(1);
  }
  console.info(`MongoDB connected (database: ${mongoose.connection.name})`);

  const app = createApp({
    modelClient: createModelClient({
      baseUrl: config.modelServiceUrl,
      timeoutMs: config.modelServiceTimeoutMs,
    }),
    corsOrigin: config.corsOrigin,
    rateLimitPerMinute: config.rateLimitPerMinute,
  });
  const server = app.listen(config.port, () =>
    console.info(`API listening on http://localhost:${config.port}`),
  );

  const shutdown = async () => {
    server.close();
    await mongoose.disconnect();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
