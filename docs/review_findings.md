# Updated source and artifact review

## Included and checked
- SHA256 of all 63 files in the supplemental upload matched its collection manifest.
- Original two-chain train/validation/test datasets and relevance diagnostic are bundled.
- Supplied token-audit JSON exists for Coconut, Huginn and Ouro.
- SIM-CoT wrapper and eval YAML are bundled with the supplied Apache 2.0 licence.
- Saved 40-group trace summaries exist for all four model families.
- RQ2 main run configuration and saved summaries are bundled.
- RBS dataset, 40-graph node-readout tables and order-run summaries are bundled.
- Uploaded experiment source bytes remain unchanged, indexed in source_manifest.json.
- Solver audits passed for all 3,120 supplied synthetic examples.
- Exact questions do not overlap between the original three splits; graph
  isomorphism separation is not claimed.
- RQ2 generated dataset bytes reproduce the recorded train/development hashes.
- RBS solver audit passes for 419 records, including 217 four-step-eligible graphs.
- No GPU model execution or numerical prediction regeneration was performed.

## Still needed for complete cross-machine reproduction
- Original local GPT-2 files and checkpoints, external to GitHub.
- Huginn torch24 compatibility model code. The supplemental upload contains it,
  but no model-code licence was supplied, so this distribution records metadata
  and leaves that code external pending licence verification.
- Ouro local model code remains external for the same licensing review.
- Scripts still contain original lab paths and output conventions.
- Complete four-model combined analysis and plotting pipeline.
- Some RBS order-control and RQ2 intervention CSVs/prediction-level files.
- SIM-CoT relevance token audit, not supplied.
- Exact environment lockfiles and NVIDIA driver record.
- Confirm repository reuse licence and model/dataset terms before public release.

## Important distinctions
- Missing SIM-CoT config.json is expected for this raw state-dict loading path:
  the script reads GPT-2 config from the external base-model directory.
- The RQ2 recorded run used eval_every=200. Reproduction instructions now include it.
- An early RQ2 diagnostic subset and a full diagnostic run are both included.
- Both SIM-CoT full relevance-run summaries are preserved; summary timestamps
  alone are insufficient to select final traces.
- The Huginn seed summaries show runs exist, not that pooled robustness was analyzed.
- The old analyze_original_model_states.py covers Coconut and Huginn only.
- Newly authored research code remains unlicensed until ownership/licence is confirmed.
