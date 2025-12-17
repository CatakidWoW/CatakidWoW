import { describe, expect, test } from 'vitest';

import { evaluateCandidate } from '../src/strategy/evaluate.js';
import type { StrategyConfig } from '../src/strategy/types.js';

const cfg: StrategyConfig = {
  minLiquidityUsd: 20000,
  minVolume24hUsd: 25000,
  maxTop10ShareBps: 5000,
  requireMintAuthorityRevoked: true,
  requireFreezeAuthorityRevoked: true,
  takeProfitMultiples: [2],
  trailingStopPct: 25,
  hardStopLossPct: 40,
  timeStopMinutes: 120
};

describe('evaluateCandidate', () => {
  test('strong when healthy metrics + authorities revoked', () => {
    const decision = evaluateCandidate(
      {
        pair: {
          source: 'dexscreener',
          chain: 'solana',
          mint: 'SomeMint',
          priceUsd: 0.01,
          liquidityUsd: 100000,
          volume24hUsd: 200000,
          priceChange24hPct: 15
        },
        onchain: {
          chain: 'solana',
          mint: 'SomeMint',
          fetchedAt: new Date().toISOString(),
          mintInfo: {
            decimals: 6,
            supply: '1000000',
            mintAuthority: null,
            freezeAuthority: null,
            isInitialized: true,
            holderConcentration: {
              top10Amount: '200000',
              top10ShareBps: 2000,
              top20Amount: '300000',
              top20ShareBps: 3000,
              largestAccounts: []
            }
          },
          metadata: { exists: true, name: 'Test', symbol: 'TST', uri: 'https://example.com' },
          score: { score: 100, issues: [] }
        }
      },
      cfg
    );

    expect(decision.tier).toBe('strong');
    expect(decision.exitPlan.entryPriceUsd).toBe(0.01);
    expect(decision.exitPlan.takeProfitPriceUsd?.[0]).toBeCloseTo(0.02);
    expect(decision.exitPlan.hardStopPriceUsd).toBeCloseTo(0.006);
  });

  test('avoid when low liquidity/volume and mint authority present', () => {
    const decision = evaluateCandidate(
      {
        pair: {
          source: 'dexscreener',
          chain: 'solana',
          mint: 'SomeMint',
          priceUsd: 1,
          liquidityUsd: 1000,
          volume24hUsd: 1000
        },
        onchain: {
          chain: 'solana',
          mint: 'SomeMint',
          fetchedAt: new Date().toISOString(),
          mintInfo: {
            decimals: 6,
            supply: '1000000',
            mintAuthority: 'Auth',
            freezeAuthority: null,
            isInitialized: true
          },
          metadata: { exists: false },
          score: { score: 0, issues: [] }
        }
      },
      cfg
    );

    expect(decision.tier).toBe('avoid');
    expect(decision.reasons.length).toBeGreaterThan(0);
  });
});
