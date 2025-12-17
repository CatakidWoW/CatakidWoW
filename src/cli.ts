import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { Connection } from '@solana/web3.js';

import { analyzeMint } from './analyze/analyzeMint.js';
import { streamNewMints } from './scan/streamNewMints.js';
import { parseRpcUrl } from './util/config.js';
import { searchSolanaPairs, pickBestSolanaPair, toSolanaPairMetrics } from './providers/dexscreener.js';
import { loadStrategyConfig } from './strategy/loadConfig.js';
import { evaluateCandidate } from './strategy/evaluate.js';
import { promisePool } from './util/promisePool.js';
import { fetchGmgnCandidates } from './providers/gmgn.js';
import { fetchAxiomTradeCandidates } from './providers/axiomTrade.js';

async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName('solana-memecoin-scanner')
    .option('rpc', {
      type: 'string',
      describe: 'Solana RPC URL (https://... or wss://...)',
      default: process.env.SOLANA_RPC_URL ?? 'https://api.mainnet-beta.solana.com'
    })
    .option('config', {
      type: 'string',
      describe: 'Path to strategy config JSON (optional)'
    })
    .command(
      'analyze <mint>',
      'Analyze a token mint address',
      (y) =>
        y.positional('mint', {
          type: 'string',
          describe: 'Token mint address'
        }),
      async (args) => {
        const rpc = parseRpcUrl(args.rpc);
        const conn = new Connection(rpc.http, { commitment: 'confirmed' });
        const report = await analyzeMint(conn, String(args.mint));
        process.stdout.write(JSON.stringify(report, null, 2) + '\n');
      }
    )
    .command(
      'scan <query>',
      'Scan Solana memecoin candidates from a source, then tier them using your strategy config',
      (y) =>
        y
          .positional('query', { type: 'string', describe: 'Search query (symbol/name/mint)' })
          .option('source', {
            type: 'string',
            default: 'dexscreener',
            choices: ['dexscreener', 'gmgn', 'axiom'] as const,
            describe: 'Data source'
          })
          .option('limit', { type: 'number', default: 10, describe: 'Max unique mints to score' })
          .option('onchain', { type: 'boolean', default: true, describe: 'Also run on-chain mint analysis' })
          .option('concurrency', { type: 'number', default: 3, describe: 'On-chain analyze concurrency' }),
      async (args) => {
        const cfg = await loadStrategyConfig(args.config as any);
        const rpc = parseRpcUrl(args.rpc);
        const conn = new Connection(rpc.http, { commitment: 'confirmed' });

        if (args.source === 'gmgn') await fetchGmgnCandidates();
        if (args.source === 'axiom') await fetchAxiomTradeCandidates();

        const pairs = await searchSolanaPairs(String(args.query));
        const byMint = new Map<string, typeof pairs>();
        for (const p of pairs) {
          const mint = p.baseToken?.address;
          if (!mint) continue;
          const arr = byMint.get(mint) ?? [];
          arr.push(p);
          byMint.set(mint, arr);
        }

        const mints = Array.from(byMint.keys()).slice(0, Number(args.limit));
        const bestPairs = mints
          .map((m) => pickBestSolanaPair(byMint.get(m) ?? []))
          .filter((p): p is NonNullable<typeof p> => Boolean(p))
          .map((p) => toSolanaPairMetrics(p));

        const withOnchain = Boolean(args.onchain);
        const candidates = withOnchain
          ? await promisePool(bestPairs, Number(args.concurrency), async (pair) => {
              const onchain = pair.mint ? await analyzeMint(conn, pair.mint) : undefined;
              const decision = evaluateCandidate({ pair, onchain }, cfg);
              return { pair, decision, onchain };
            })
          : bestPairs.map((pair) => {
              const decision = evaluateCandidate({ pair }, cfg);
              return { pair, decision };
            });

        // Sort by tier then liquidity as a tie-breaker.
        const tierRank: Record<string, number> = { strong: 3, speculative: 2, watch: 1, avoid: 0 };
        candidates.sort((a, b) => {
          const ta = tierRank[a.decision.tier] ?? 0;
          const tb = tierRank[b.decision.tier] ?? 0;
          if (tb !== ta) return tb - ta;
          const la = a.pair.liquidityUsd ?? 0;
          const lb = b.pair.liquidityUsd ?? 0;
          return lb - la;
        });

        process.stdout.write(JSON.stringify({ query: args.query, config: cfg, results: candidates }, null, 2) + '\n');
      }
    )
    .command(
      'stream',
      'Watch for new SPL token mints (logs subscription) and analyze them',
      (y) =>
        y
          .option('commitment', {
            type: 'string',
            default: 'confirmed',
            choices: ['processed', 'confirmed', 'finalized'] as const
          })
          .option('max', {
            type: 'number',
            default: 25,
            describe: 'Stop after analyzing this many newly detected mints'
          }),
      async (args) => {
        const rpc = parseRpcUrl(args.rpc);
        const conn = new Connection(rpc.http, { commitment: args.commitment as any, wsEndpoint: rpc.ws });
        await streamNewMints(conn, {
          maxToAnalyze: Number(args.max)
        });
      }
    )
    .demandCommand(1)
    .strict()
    .help();

  await argv.parseAsync();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
