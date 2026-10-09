/** Plain-language column titles. Anything not listed falls back to "Sentence case". */
const HEADERS: Record<string, string> = {
  id: 'ID', run_id: 'Run', attempt_id: 'Attempt', objective_id: 'Objective', cipher_type: 'Cipher', cipher_key_id: 'Key fingerprint',
  strategy_name: 'Strategy', defense_name: 'Defense', defense: 'Defense', iteration: 'Try #', status: 'Result', created_at: 'When',
  target_model: 'Model', target_response: 'Model reply', plaintext_prompt: 'Original prompt', ciphertext_payload: 'Encoded prompt',
  decode_instruction: 'Decode hint',
  full_request_sent: 'Exact request sent to model', reproduction_steps: 'How to reproduce this attack',
  plaintext_size_bytes: 'Prompt size (B)', ciphertext_size_bytes: 'Encoded size (B)',
  encryption_time_ms: 'Encode time (ms)', decryption_time_ms: 'Decode time (ms)', avalanche_score: 'Avalanche', confusion_chi2: 'Chi-square',
  diffusion_score: 'Diffusion', shannon_entropy: 'Entropy (bits)', latency_ms: 'Reply time (ms)', prompt_tokens: 'Tokens in',
  completion_tokens: 'Tokens out', cost_usd: 'Cost (USD)', filter_evaded: 'Got past filter', semantic_complied: 'Followed the request',
  harm_score_0_1: 'Harm score', judge_rationale: 'Why', judge_model: 'Judge', classifier_label: 'Safety label', judge_lm_verdict: 'Judge says yes',
  llamaguard_verdict: 'Llama Guard says unsafe', regex_verdict: 'Canary found', score_source: 'Scored by', blocked: 'Blocked',
  transformed_payload: 'Changed prompt', defense_name_: 'Defense', dataset_name: 'Dataset', seed: 'Seed', budget_usd: 'Budget (USD)',
  spent_usd: 'Spent (USD)', max_iterations: 'Max tries', owasp_tag: 'OWASP tag', atlas_tag: 'ATLAS tag', config_json: 'Settings',
  attacker_provider_id: 'Attacker', target_provider_id: 'Target', judge_provider_id: 'Judge', report_hash: 'Report hash',
  item_count: 'Items', hold_until: 'Hold until', vendor_name: 'Vendor', generated_at: 'Created', last_verified_at: 'Last checked',
}
export const headerLabel = (k: string) => HEADERS[k] ?? (k.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()))

