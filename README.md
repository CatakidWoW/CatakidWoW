# Solana Memecoin Scanner (Solana-only)

CLI tool that:

- **Streams** newly created SPL token mints on Solana (via Token Program logs), then auto-analyzes each mint.
- **Analyzes** any mint address to produce a JSON “memecoin risk” report (authorities, metadata, holder concentration).

## Install

```bash
npm i
```

## Configure RPC

Use `SOLANA_RPC_URL` or pass `--rpc`.

Examples:

```bash
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
```

## Usage

### Analyze a mint (one-off)

```bash
npm run dev -- analyze <MINT_ADDRESS>
```

### Scan candidates via DexScreener (then tier them)

```bash
npm run dev -- scan "bonk" --limit 10
```

To disable on-chain checks (faster, less RPC usage):

```bash
npm run dev -- scan "bonk" --onchain false
```

Select a source (currently only DexScreener is implemented):

```bash
npm run dev -- scan "bonk" --source dexscreener
```

### Strategy config

Optionally pass a JSON config file:

```bash
npm run dev -- scan "bonk" --config ./my-strategy.json
```

### Stream newly created mints (and analyze each)

```bash
npm run dev -- stream --max 25
```

## Output

Prints JSON to stdout per mint:

- `mintInfo`: decimals, supply, mint/freeze authority, basic holder concentration (top 10/20 accounts)
- `metadata`: Metaplex name/symbol/uri (+ tries to fetch JSON if URI is HTTP/S)
- `score`: simple 0–100 score plus issues

## Notes on gmgn.ai / Axiom

This tool uses **DexScreener** because it has a stable public JSON API. If you want gmgn.ai or Axiom integration, we should use an official API or permissioned endpoint (scraping sites is fragile and may violate terms).
