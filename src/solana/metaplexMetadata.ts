import { Connection, PublicKey } from '@solana/web3.js';
import fetch from 'node-fetch';

const METADATA_PROGRAM_ID = new PublicKey('metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s');

export type MetaplexMetadata = {
  exists: boolean;
  name?: string;
  symbol?: string;
  uri?: string;
  uriJson?: any;
  error?: string;
};

function readFixedString(buf: Buffer, offset: number, length: number): string {
  const slice = buf.subarray(offset, offset + length);
  const nul = slice.indexOf(0);
  return slice.subarray(0, nul === -1 ? length : nul).toString('utf8').trim();
}

/**
 * Minimal decoder for Metaplex Token Metadata v1 Data fields.
 * Layout: key(1) + updateAuth(32) + mint(32) + name(4+32) + symbol(4+10) + uri(4+200) ...
 * We only need name/symbol/uri, so this keeps dependencies small.
 */
function decodeNameSymbolUri(accountData: Buffer): { name: string; symbol: string; uri: string } {
  // Skip: key(1) + updateAuthority(32) + mint(32)
  let o = 1 + 32 + 32;

  const nameLen = accountData.readUInt32LE(o);
  o += 4;
  const name = accountData.subarray(o, o + Math.min(nameLen, 32)).toString('utf8').replace(/\0/g, '').trim();
  o += 32;

  const symLen = accountData.readUInt32LE(o);
  o += 4;
  const symbol = accountData.subarray(o, o + Math.min(symLen, 10)).toString('utf8').replace(/\0/g, '').trim();
  o += 10;

  const uriLen = accountData.readUInt32LE(o);
  o += 4;
  const uri = accountData.subarray(o, o + Math.min(uriLen, 200)).toString('utf8').replace(/\0/g, '').trim();

  // If above fails due to unexpected layout, fallback to fixed offsets common in practice.
  if (!name && !symbol && !uri) {
    // This fallback is best-effort.
    const fallbackName = readFixedString(accountData, 1 + 32 + 32 + 4, 32);
    const fallbackSymbol = readFixedString(accountData, 1 + 32 + 32 + 4 + 32 + 4, 10);
    const fallbackUri = readFixedString(accountData, 1 + 32 + 32 + 4 + 32 + 4 + 10 + 4, 200);
    return { name: fallbackName, symbol: fallbackSymbol, uri: fallbackUri };
  }

  return { name, symbol, uri };
}

async function tryFetchJson(uri: string): Promise<any> {
  // Many tokens use arweave/ipfs gateways; fetch if it looks like HTTP(s)
  if (!/^https?:\/\//i.test(uri)) return undefined;
  const res = await fetch(uri, { redirect: 'follow' });
  if (!res.ok) return undefined;
  const ct = res.headers.get('content-type') ?? '';
  if (!ct.includes('application/json') && !ct.includes('text/plain')) return undefined;
  return await res.json().catch(() => undefined);
}

export async function fetchMetaplexMetadata(conn: Connection, mint: PublicKey): Promise<MetaplexMetadata> {
  try {
    const [pda] = PublicKey.findProgramAddressSync(
      [Buffer.from('metadata'), METADATA_PROGRAM_ID.toBuffer(), mint.toBuffer()],
      METADATA_PROGRAM_ID
    );

    const ai = await conn.getAccountInfo(pda, 'confirmed');
    if (!ai?.data) return { exists: false };

    const buf = Buffer.from(ai.data);
    const { name, symbol, uri } = decodeNameSymbolUri(buf);

    const uriJson = uri ? await tryFetchJson(uri) : undefined;

    return {
      exists: true,
      name,
      symbol,
      uri,
      uriJson
    };
  } catch (e: any) {
    return { exists: false, error: String(e?.message ?? e) };
  }
}
