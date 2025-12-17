import { describe, expect, test } from 'vitest';
import { computeScore } from '../src/analyze/score.js';

describe('computeScore', () => {
  test('penalizes mint authority and freeze authority', () => {
    const res = computeScore({
      mintInfo: {
        decimals: 6,
        supply: '1000000',
        mintAuthority: 'SomeAuth',
        freezeAuthority: 'SomeFreeze',
        isInitialized: true,
        holderConcentration: {
          top10Amount: '800000',
          top10ShareBps: 8000,
          top20Amount: '900000',
          top20ShareBps: 9000,
          largestAccounts: []
        }
      },
      metadata: { exists: false }
    });

    expect(res.score).toBeLessThan(60);
    expect(res.issues.some((i) => i.code === 'MINT_AUTHORITY_PRESENT')).toBe(true);
    expect(res.issues.some((i) => i.code === 'FREEZE_AUTHORITY_PRESENT')).toBe(true);
  });

  test('gives higher score when authorities revoked and metadata present', () => {
    const res = computeScore({
      mintInfo: {
        decimals: 9,
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
      metadata: { exists: true, name: 'Test', symbol: 'TST', uri: 'https://example.com/meta.json' }
    });

    expect(res.score).toBeGreaterThan(80);
  });
});
