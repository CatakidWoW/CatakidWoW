export type AxiomTradeCandidate = {
  // Placeholder: Axiom Trade integration requires an official API/permissioned endpoint.
  source: 'axiom';
};

export async function fetchAxiomTradeCandidates(): Promise<AxiomTradeCandidate[]> {
  throw new Error('Axiom provider not implemented: add an official API endpoint / auth first.');
}
