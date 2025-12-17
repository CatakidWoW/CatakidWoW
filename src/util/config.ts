export type RpcUrl = { http: string; ws?: string };

export function parseRpcUrl(input: unknown): RpcUrl {
  const raw = String(input ?? '').trim();
  if (!raw) throw new Error('RPC URL is empty');

  // If user passes wss://..., derive https://... for HTTP calls.
  if (raw.startsWith('wss://') || raw.startsWith('ws://')) {
    const http = raw.replace(/^wss?:\/\//, (m) => (m === 'wss://' ? 'https://' : 'http://'));
    return { http, ws: raw };
  }

  // If user passes https://..., derive wss://... for websockets.
  if (raw.startsWith('https://') || raw.startsWith('http://')) {
    const ws = raw.replace(/^https?:\/\//, (m) => (m === 'https://' ? 'wss://' : 'ws://'));
    return { http: raw, ws };
  }

  throw new Error(`Unsupported RPC URL scheme: ${raw}`);
}
