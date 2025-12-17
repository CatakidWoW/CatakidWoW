import type { StrategyConfig, Candidate, Decision, Tier } from './types.js';

function pctToPrice(price: number, pct: number): number {
  return price * (1 - pct / 100);
}

function tierFromScore(score: number): Tier {
  if (score >= 85) return 'strong';
  if (score >= 70) return 'speculative';
  if (score >= 55) return 'watch';
  return 'avoid';
}

export function evaluateCandidate(candidate: Candidate, cfg: StrategyConfig): Decision {
  const reasons: string[] = [];

  // Start from 100 and subtract for risk.
  let s = 100;

  const p = candidate.pair;
  const liq = p.liquidityUsd ?? 0;
  const vol = p.volume24hUsd ?? 0;

  if (liq < cfg.minLiquidityUsd) {
    s -= 25;
    reasons.push(`Low liquidity: $${liq.toFixed(0)} < $${cfg.minLiquidityUsd}`);
  }

  if (vol < cfg.minVolume24hUsd) {
    s -= 15;
    reasons.push(`Low 24h volume: $${vol.toFixed(0)} < $${cfg.minVolume24hUsd}`);
  }

  if (typeof p.priceChange24hPct === 'number' && p.priceChange24hPct > 300) {
    s -= 10;
    reasons.push(`Very high 24h pump (+${p.priceChange24hPct}%)`);
  }

  // On-chain checks (if we have a mint)
  const oc = candidate.onchain;
  if (oc) {
    if (cfg.requireMintAuthorityRevoked && oc.mintInfo.mintAuthority) {
      s -= 25;
      reasons.push('Mint authority still enabled (can inflate supply)');
    }
    if (cfg.requireFreezeAuthorityRevoked && oc.mintInfo.freezeAuthority) {
      s -= 20;
      reasons.push('Freeze authority still enabled (can freeze holders)');
    }

    const top10 = oc.mintInfo.holderConcentration?.top10ShareBps;
    if (typeof top10 === 'number' && top10 > cfg.maxTop10ShareBps) {
      s -= 20;
      reasons.push(`Top10 concentration too high: ${(top10 / 100).toFixed(2)}% > ${(cfg.maxTop10ShareBps / 100).toFixed(2)}%`);
    }

    if (!oc.metadata.exists) {
      s -= 5;
      reasons.push('No Metaplex metadata found');
    }
  } else {
    s -= 5;
    if (candidate.pair.mint) {
      reasons.push('On-chain mint analysis not included (disabled or unavailable)');
    } else {
      reasons.push('On-chain mint analysis not available (no mint address)');
    }
  }

  // Clamp
  if (s < 0) s = 0;
  if (s > 100) s = 100;

  const tier = tierFromScore(s);

  const entry = p.priceUsd;
  const takeProfitPriceUsd = typeof entry === 'number' && entry > 0
    ? cfg.takeProfitMultiples.map((m) => entry * m)
    : undefined;

  const hardStopPriceUsd = typeof entry === 'number' && entry > 0
    ? pctToPrice(entry, cfg.hardStopLossPct)
    : undefined;

  return {
    tier,
    reasons,
    exitPlan: {
      entryPriceUsd: entry,
      takeProfitPriceUsd,
      hardStopPriceUsd,
      trailingStopPct: cfg.trailingStopPct,
      timeStopMinutes: cfg.timeStopMinutes
    }
  };
}
