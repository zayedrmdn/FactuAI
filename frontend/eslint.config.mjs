import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { FlatCompat } from '@eslint/eslintrc';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  // 1. Specify folders to ignore first
  {
    ignores: [
      ".next/*",
      "node_modules/*",
      "dist/*",
      "build/*",
      "public/*",
      "out/*",
    ],
  },
  // 2. Spread the existing Next.js configs
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  // 3. Add custom rules or overrides here if needed
  {
    rules: {
      // Example: "no-unused-vars": "warn"
    },
  },
];

export default eslintConfig;