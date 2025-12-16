import { Connection, PublicKey } from '@solana/web3.js';

import { fetchTokenMintInfo } from './fetchTokenMintInfo.js';
import { fetchMetaplexMetadata } from '../solana/metaplexMetadata.js';
import { computeScore, type ScoreResult } from './score.js';

export type MintAnalysis = {
  chain: 'solana';
  mint: string;
  fetchedAt: string;
  mintInfo: Awaited<ReturnType<typeof fetchTokenMintInfo>>;
  metadata: Awaited<ReturnType<typeof fetchMetaplexMetadata>>;
  score: ScoreResult;
};

export async function analyzeMint(conn: Connection, mintAddress: string): Promise<MintAnalysis> {
  const mintPk = new PublicKey(mintAddress);

  const [mintInfo, metadata] = await Promise.all([
    fetchTokenMintInfo(conn, mintPk),
    fetchMetaplexMetadata(conn, mintPk)
  ]);

  const score = computeScore({ mintInfo, metadata });

  return {
    chain: 'solana',
    mint: mintPk.toBase58(),
    fetchedAt: new Date().toISOString(),
    mintInfo,
    metadata,
    score
  };
}
