import type { SolanaPairMetrics } from '../providers/dexscreener.js';
import type { MintAnalysis } from '../analyze/analyzeMint.js';

export type Tier = 'avoid' | 'watch' | 'speculative' | 'strong';

export type StrategyConfig = {
  // Liquidity/volume safety filters
  minLiquidityUsd: number;
  minVolume24hUsd: number;

  // Concentration filters (basis points)
  maxTop10ShareBps: number; // e.g. 4500 = 45%

  // Authority safety
  requireMintAuthorityRevoked: boolean;
  requireFreezeAuthorityRevoked: boolean;

  // Exit plan (rule-based, not predictive)
  takeProfitMultiples: number[]; // e.g. [1.5, 2, 3]
  trailingStopPct: number; // e.g. 25 means trail 25%
  hardStopLossPct: number; // e.g. 35 means stop out at -35%
  timeStopMinutes?: number; // optional
};

export type Decision = {
  tier: Tier;
  reasons: string[];
  exitPlan: {
    entryPriceUsd?: number;
    takeProfitPriceUsd?: number[];
    hardStopPriceUsd?: number;
    trailingStopPct: number;
    timeStopMinutes?: number;
  };
};

export type Candidate = {
  pair: SolanaPairMetrics;
  onchain?: MintAnalysis;
};
