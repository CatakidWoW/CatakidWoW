import type { TokenMintInfo } from './fetchTokenMintInfo.js';
import type { MetaplexMetadata } from '../solana/metaplexMetadata.js';

export type ScoreIssue = {
  code: string;
  severity: 'low' | 'medium' | 'high';
  message: string;
};

export type ScoreResult = {
  score: number; // 0..100 (higher = better)
  issues: ScoreIssue[];
};

export type ScoreInput = {
  mintInfo: TokenMintInfo;
  metadata: MetaplexMetadata;
};

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

export function computeScore(input: ScoreInput): ScoreResult {
  const issues: ScoreIssue[] = [];
  let score = 100;

  // Authorities
  if (input.mintInfo.mintAuthority) {
    score -= 25;
    issues.push({
      code: 'MINT_AUTHORITY_PRESENT',
      severity: 'high',
      message: 'Mint authority is still enabled (supply can be inflated).'
    });
  }
  if (input.mintInfo.freezeAuthority) {
    score -= 15;
    issues.push({
      code: 'FREEZE_AUTHORITY_PRESENT',
      severity: 'high',
      message: 'Freeze authority is still enabled (accounts can be frozen).'
    });
  }

  // Metadata
  if (!input.metadata.exists) {
    score -= 10;
    issues.push({
      code: 'NO_METADATA',
      severity: 'medium',
      message: 'No Metaplex metadata account found.'
    });
  } else {
    if (!input.metadata.name || input.metadata.name.length < 2) {
      score -= 3;
      issues.push({
        code: 'WEAK_NAME',
        severity: 'low',
        message: 'Token name is missing/very short.'
      });
    }
    if (!input.metadata.symbol || input.metadata.symbol.length < 2) {
      score -= 3;
      issues.push({
        code: 'WEAK_SYMBOL',
        severity: 'low',
        message: 'Token symbol is missing/very short.'
      });
    }
    if (!input.metadata.uri) {
      score -= 4;
      issues.push({
        code: 'NO_URI',
        severity: 'medium',
        message: 'Metadata URI is missing.'
      });
    }
  }

  // Holder concentration (very rough heuristic)
  const hc = input.mintInfo.holderConcentration;
  if (hc) {
    if (hc.top10ShareBps >= 5000) {
      score -= 20;
      issues.push({
        code: 'TOP10_CONCENTRATED',
        severity: 'high',
        message: `Top 10 token accounts hold ${hc.top10ShareBps / 100}% of supply (high concentration).`
      });
    } else if (hc.top10ShareBps >= 3000) {
      score -= 10;
      issues.push({
        code: 'TOP10_SOMEWHAT_CONCENTRATED',
        severity: 'medium',
        message: `Top 10 token accounts hold ${hc.top10ShareBps / 100}% of supply.`
      });
    }
  } else {
    score -= 3;
    issues.push({
      code: 'NO_HOLDER_DATA',
      severity: 'low',
      message: 'Could not fetch holder concentration (RPC limitations or token not widely held yet).'
    });
  }

  return {
    score: clamp(score, 0, 100),
    issues
  };
}
