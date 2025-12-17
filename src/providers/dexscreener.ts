import fetch from 'node-fetch';

export type DexScreenerPair = {
  chainId: string;
  dexId?: string;
  url?: string;
  pairAddress?: string;
  baseToken?: { address?: string; symbol?: string; name?: string };
  quoteToken?: { address?: string; symbol?: string; name?: string };
  priceUsd?: string;
  fdv?: number;
  marketCap?: number;
  liquidity?: { usd?: number };
  volume?: { h24?: number; h6?: number; h1?: number; m5?: number };
  priceChange?: { h24?: number; h6?: number; h1?: number; m5?: number };
  txns?: {
    h24?: { buys?: number; sells?: number };
    h6?: { buys?: number; sells?: number };
    h1?: { buys?: number; sells?: number };
    m5?: { buys?: number; sells?: number };
  };
  pairCreatedAt?: number;
};

export type DexScreenerSearchResponse = { pairs?: DexScreenerPair[] };
export type DexScreenerTokenResponse = { pairs?: DexScreenerPair[] };

function toNum(x: unknown): number | undefined {
  if (typeof x === 'number' && Number.isFinite(x)) return x;
  if (typeof x === 'string' && x.trim() !== '') {
    const n = Number(x);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

export type SolanaPairMetrics = {
  source: 'dexscreener';
  chain: 'solana';
  pairAddress?: string;
  mint?: string; // base token address
  symbol?: string;
  name?: string;
  priceUsd?: number;
  liquidityUsd?: number;
  fdv?: number;
  marketCap?: number;
  volume24hUsd?: number;
  priceChange24hPct?: number;
  buys24h?: number;
  sells24h?: number;
  pairCreatedAt?: number;
  url?: string;
};

export function pickBestSolanaPair(pairs: DexScreenerPair[]): DexScreenerPair | undefined {
  const sol = pairs.filter((p) => p.chainId === 'solana');
  // Prefer highest liquidity, fallback to volume.
  return sol.sort((a, b) => {
    const la = a.liquidity?.usd ?? 0;
    const lb = b.liquidity?.usd ?? 0;
    if (lb !== la) return lb - la;
    const va = a.volume?.h24 ?? 0;
    const vb = b.volume?.h24 ?? 0;
    return vb - va;
  })[0];
}

export function toSolanaPairMetrics(pair: DexScreenerPair): SolanaPairMetrics {
  return {
    source: 'dexscreener',
    chain: 'solana',
    pairAddress: pair.pairAddress,
    mint: pair.baseToken?.address,
    symbol: pair.baseToken?.symbol,
    name: pair.baseToken?.name,
    priceUsd: toNum(pair.priceUsd),
    liquidityUsd: pair.liquidity?.usd,
    fdv: pair.fdv,
    marketCap: pair.marketCap,
    volume24hUsd: pair.volume?.h24,
    priceChange24hPct: pair.priceChange?.h24,
    buys24h: pair.txns?.h24?.buys,
    sells24h: pair.txns?.h24?.sells,
    pairCreatedAt: pair.pairCreatedAt,
    url: pair.url
  };
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    headers: {
      'accept': 'application/json'
    }
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`DexScreener HTTP ${res.status}: ${txt.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

/** Search by symbol/name/address and return matching Solana pairs. */
export async function searchSolanaPairs(query: string): Promise<DexScreenerPair[]> {
  const q = encodeURIComponent(query.trim());
  const url = `https://api.dexscreener.com/latest/dex/search?q=${q}`;
  const json = await getJson<DexScreenerSearchResponse>(url);
  return (json.pairs ?? []).filter((p) => p.chainId === 'solana');
}

/** Fetch pairs by token address and return Solana pairs. */
export async function getSolanaPairsByToken(tokenAddress: string): Promise<DexScreenerPair[]> {
  const t = encodeURIComponent(tokenAddress.trim());
  const url = `https://api.dexscreener.com/latest/dex/tokens/${t}`;
  const json = await getJson<DexScreenerTokenResponse>(url);
  return (json.pairs ?? []).filter((p) => p.chainId === 'solana');
}
