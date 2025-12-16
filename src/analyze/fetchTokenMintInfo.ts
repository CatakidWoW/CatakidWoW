import { Connection, PublicKey } from '@solana/web3.js';
import { getMint } from '@solana/spl-token';

export type HolderConcentration = {
  top10Amount: string;
  top10ShareBps: number; // 0..10000
  top20Amount: string;
  top20ShareBps: number;
  largestAccounts: { address: string; amount: string }[];
};

export type TokenMintInfo = {
  decimals: number;
  supply: string;
  mintAuthority: string | null;
  freezeAuthority: string | null;
  isInitialized: boolean;
  holderConcentration?: HolderConcentration;
};

function toBps(numerator: bigint, denominator: bigint): number {
  if (denominator === 0n) return 0;
  const bps = (numerator * 10000n) / denominator;
  return Number(bps > 10000n ? 10000n : bps);
}

export async function fetchTokenMintInfo(conn: Connection, mint: PublicKey): Promise<TokenMintInfo> {
  const m = await getMint(conn, mint, 'confirmed');

  const supply = m.supply;
  const mintAuthority = m.mintAuthority?.toBase58() ?? null;
  const freezeAuthority = m.freezeAuthority?.toBase58() ?? null;

  // Holder distribution: largest token accounts.
  // Note: this is approximate for holder concentration; excludes off-chain/DEX LP ownership semantics.
  let holderConcentration: HolderConcentration | undefined;
  try {
    const largest = await conn.getTokenLargestAccounts(mint, 'confirmed');
    const largestAccounts = largest.value.map((v) => ({ address: v.address.toBase58(), amount: v.amount }));

    const top10 = largestAccounts.slice(0, 10).reduce((acc, x) => acc + BigInt(x.amount), 0n);
    const top20 = largestAccounts.slice(0, 20).reduce((acc, x) => acc + BigInt(x.amount), 0n);

    holderConcentration = {
      top10Amount: top10.toString(),
      top10ShareBps: toBps(top10, supply),
      top20Amount: top20.toString(),
      top20ShareBps: toBps(top20, supply),
      largestAccounts
    };
  } catch {
    // ignore
  }

  return {
    decimals: m.decimals,
    supply: supply.toString(),
    mintAuthority,
    freezeAuthority,
    isInitialized: m.isInitialized,
    holderConcentration
  };
}
