# Reproduction

## Environment
The recorded primary stack is Python 3.10, PyTorch 2.4.1+cu121 and
Transformers 4.46.2. Ouro tracing used Transformers 4.55.0 in a separate
environment. Environment files in environments are minimal specifications,
not exported lockfiles or a fresh-install guarantee.

## Original-checkpoint tracing on the existing server
The original scripts rely on the working directory and in some cases absolute
/mnt/home/user16/nihad paths. Run them from the existing mentor_experiment
folder and point Python at the repository script:
```bash
cd /mnt/home/user16/nihad/coconut/mentor_experiment
python /path/to/latent-reasoning-study/experiments/original_checkpoint_states/trace_mentor_coconut_layers.py --groups 2
python /path/to/latent-reasoning-study/experiments/original_checkpoint_states/trace_original_huginn_iterations.py --groups 2
python /path/to/latent-reasoning-study/experiments/relevance_controls/audit_control_tokens.py --model coconut
python /path/to/latent-reasoning-study/experiments/relevance_controls/trace_coconut_relevance_control.py --groups 2
```
Read each script's paths before running. These commands need existing models,
data and wrappers. Do not move the original lab directory to match this package.
For full runs use 40 groups after the smoke audit passes.

SIM-CoT exposes --root for the nihad folder. It additionally checks a specific
wrapper hash and evaluation YAML. Reference copies are now bundled under third_party/simcot_reference. Ouro requires its original local
model code and download_revision.json. Do not replace compatibility code with
unreviewed upstream code and claim identical reproduction.

## RBS
Keep the upstream checkout separately, with its dataset and checkpoint.
```bash
python experiments/rbs_node_readout/run_rbs_node_readout.py --repo-root /absolute/path/to/reasoning_by_superposition_review --graphs 40 --steps 4 --seed 2026 --bootstrap 2000
python experiments/rbs_order_controls/run_rbs_order_controls.py --repo-root /absolute/path/to/reasoning_by_superposition_review --parent-dir /absolute/path/to/node_readout_run --graphs 40
```
The second command needs the frozen parent cohort and its outputs.
The supplied third_party files are reference copies, not the complete upstream dataset.

## RQ2 diagnostics
After training with --save-weights, use the printed run directory:
```bash
python experiments/common_backbone_feasibility/diagnose_rq2_workspace_v1.py --run-dir /absolute/path/to/run --model-dir /absolute/path/to/gpt2_model --groups 0
```
The diagnostic enforces the recorded torch and transformers versions.
The older audit_rq2_existing_baselines.py is a historical implementation audit
with additional external dependencies, not the clean prototype launcher.

## Analysis scope
analyze_original_model_states.py is the original Coconut-Huginn analysis.
Its filename does not mean it is the final four-model combined analysis.
The latter script and final figure-generating pipeline are not identified in
the supplied archive and must be collected before claiming complete reproduction.

## Supplemental inputs now bundled
data/controlled_v3_twochains contains the original three splits.
data/relevance_control_20261011_171002 contains diagnostic.json and three audits.
The original scripts still expect their lab locations. On the existing server
those files already exist; do not overwrite them with a different dataset.

third_party/simcot_reference contains the reviewed wrapper and eval YAML.
The tracing scripts still resolve the original external SIM-CoT_review folder.
third_party/rbs_review now includes the supplied graph dataset, tokenizer,
wrapper and config, so it can serve as --repo-root after obtaining the
checksum-matched checkpoint under artifacts/coconut_theory/checkpoint_300.
Outputs there are ignored by .gitignore.

On this bundled dataset, the CPU-only RBS audit command is:
```bash
python experiments/rbs_node_readout/run_rbs_node_readout.py --audit-data third_party/rbs_review/data/prosqa_test_graph_4_coconut.json
```
