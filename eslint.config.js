// Flat ESLint config for the NON-BLOCKING CI lint job (see .github/workflows/ci.yml).
// Covers the hand-written JS in api/ and web/. Findings are advisory today; the
// owner can flip the lint job to required later. Tuned to the existing style —
// do NOT mass-reformat to satisfy it. Run: npx eslint api web
import js from '@eslint/js';

export default [
  {
    // Generated / vendored / non-source paths.
    ignores: ['node_modules/**', '.venv/**', 'app/**', '.vercel/**'],
  },
  {
    files: ['api/**/*.js', 'web/**/*.js'],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: 'module',
      globals: {
        // Web Platform + Vercel serverless runtime globals used across api/ and web/.
        console: 'readonly',
        process: 'readonly',
        fetch: 'readonly',
        URL: 'readonly',
        URLSearchParams: 'readonly',
        TextEncoder: 'readonly',
        TextDecoder: 'readonly',
        Response: 'readonly',
        Request: 'readonly',
        Headers: 'readonly',
        document: 'readonly',
        window: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        AbortController: 'readonly',
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      // Advisory-only relaxations tuned to the existing style; these are not bugs.
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
      'no-empty': ['warn', { allowEmptyCatch: true }],
    },
  },
];
