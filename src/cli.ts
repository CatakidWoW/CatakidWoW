import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { Connection } from '@solana/web3.js';

import { analyzeMint } from './analyze/analyzeMint.js';
import { streamNewMints } from './scan/streamNewMints.js';
import { parseRpcUrl } from './util/config.js';

async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName('solana-memecoin-scanner')
    .option('rpc', {
      type: 'string',
      describe: 'Solana RPC URL (https://... or wss://...)',
      default: process.env.SOLANA_RPC_URL ?? 'https://api.mainnet-beta.solana.com'
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
