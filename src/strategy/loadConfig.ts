import { readFile } from 'node:fs/promises';

import type { StrategyConfig } from './types.js';
import { defaultStrategyConfig } from './defaultConfig.js';

export async function loadStrategyConfig(path?: string): Promise<StrategyConfig> {
  if (!path) return defaultStrategyConfig;

  const raw = await readFile(path, 'utf8');
  const json = JSON.parse(raw) as Partial<StrategyConfig>;

  return {
    ...defaultStrategyConfig,
    ...json,
    takeProfitMultiples: json.takeProfitMultiples ?? defaultStrategyConfig.takeProfitMultiples
  };
}
