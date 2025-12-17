import { Connection, PublicKey, ParsedInstruction } from '@solana/web3.js';

import { analyzeMint } from '../analyze/analyzeMint.js';

const TOKEN_PROGRAM_ID = new PublicKey('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA');

export type StreamOptions = {
  maxToAnalyze: number;
};

function extractInitializedMintsFromParsedInstructions(ixs: ParsedInstruction[]): string[] {
  const out: string[] = [];
  for (const ix of ixs) {
    if (ix.programId?.toBase58?.() !== TOKEN_PROGRAM_ID.toBase58()) continue;
    if (ix.parsed?.type !== 'initializeMint' && ix.parsed?.type !== 'initializeMint2') continue;
    const mint = ix.parsed?.info?.mint;
    if (typeof mint === 'string') out.push(mint);
  }
  return out;
}

export async function streamNewMints(conn: Connection, opts: StreamOptions): Promise<void> {
  const seen = new Set<string>();
  let analyzed = 0;

  // Subscribe to Token Program logs; on each tx, fetch parsed tx and detect InitializeMint.
  const subId = conn.onLogs(
    TOKEN_PROGRAM_ID,
    async (logInfo) => {
      try {
        if (analyzed >= opts.maxToAnalyze) return;

        const sig = logInfo.signature;
        const tx = await conn.getParsedTransaction(sig, {
          maxSupportedTransactionVersion: 0,
          commitment: 'confirmed'
        });

        const ixs = tx?.transaction.message.instructions;
        if (!ixs) return;

        const mints = extractInitializedMintsFromParsedInstructions(ixs as ParsedInstruction[]);
        for (const mint of mints) {
          if (seen.has(mint)) continue;
          seen.add(mint);

          const report = await analyzeMint(conn, mint);
          process.stdout.write(JSON.stringify(report, null, 2) + '\n');

          analyzed += 1;
          if (analyzed >= opts.maxToAnalyze) {
            await conn.removeOnLogsListener(subId);
          }
        }
      } catch (e) {
        console.error('stream error:', e);
      }
    },
    'confirmed'
  );

  // Keep process alive until unsubscribed.
  await new Promise<void>((resolve) => {
    const t = setInterval(() => {
      if (analyzed >= opts.maxToAnalyze) {
        clearInterval(t);
        resolve();
      }
    }, 250);
  });
}
