import { describe, expect, test } from 'vitest';

import { pickBestSolanaPair, toSolanaPairMetrics } from '../src/providers/dexscreener.js';

describe('dexscreener helpers', () => {
  test('pickBestSolanaPair prefers higher liquidity', () => {
    const best = pickBestSolanaPair([
      { chainId: 'solana', liquidity: { usd: 100 }, volume: { h24: 1000 }, baseToken: { address: 'A' } },
      { chainId: 'solana', liquidity: { usd: 200 }, volume: { h24: 10 }, baseToken: { address: 'B' } },
      { chainId: 'ethereum', liquidity: { usd: 999999 }, volume: { h24: 999999 }, baseToken: { address: 'E' } }
    ] as any);

    expect(best?.baseToken?.address).toBe('B');
  });

  test('toSolanaPairMetrics maps expected fields', () => {
    const m = toSolanaPairMetrics({
      chainId: 'solana',
      baseToken: { address: 'Mint', symbol: 'SYM', name: 'Name' },
      priceUsd: '0.5',
      liquidity: { usd: 123 },
      volume: { h24: 456 },
      priceChange: { h24: -10 },
      txns: { h24: { buys: 1, sells: 2 } }
    } as any);

    expect(m.chain).toBe('solana');
    expect(m.mint).toBe('Mint');
    expect(m.priceUsd).toBe(0.5);
    expect(m.liquidityUsd).toBe(123);
    expect(m.volume24hUsd).toBe(456);
    expect(m.priceChange24hPct).toBe(-10);
    expect(m.buys24h).toBe(1);
  });
});
