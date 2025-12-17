export type GmgnCandidate = {
  // Placeholder: gmgn.ai integration requires an official API/permissioned endpoint.
  source: 'gmgn';
};

export async function fetchGmgnCandidates(): Promise<GmgnCandidate[]> {
  throw new Error('gmgn.ai provider not implemented: add an official API endpoint / auth first.');
}
