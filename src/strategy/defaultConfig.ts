import type { StrategyConfig } from './types.js';

export const defaultStrategyConfig: StrategyConfig = {
  minLiquidityUsd: 20000,
  minVolume24hUsd: 25000,
  maxTop10ShareBps: 5000,
  requireMintAuthorityRevoked: true,
  requireFreezeAuthorityRevoked: true,
  takeProfitMultiples: [1.5, 2, 3],
  trailingStopPct: 25,
  hardStopLossPct: 35,
  timeStopMinutes: 240
};
