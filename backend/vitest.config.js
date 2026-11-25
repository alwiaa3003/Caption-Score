import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    testTimeout: 30_000,
    hookTimeout: 120_000, // first run of mongodb-memory-server may download a MongoDB binary
  },
});
