# Sequential Expansion vs. Looped Refinement in Latent Reasoning

Research code prepared for Md Nihad's Master's thesis at Harbin Institute
of Technology, Shenzhen, supervised by Prof. Kehai Chen.

This project studies how latent computation represents and updates information.
Controlled graphs are the experimental testbed. The repository separates
released-checkpoint observations from controlled common-backbone experiments.

## Experiments
| Folder | Purpose | Reproduction status |
| --- | --- | --- |
| original_checkpoint_states | Trace Coconut, SIM-CoT, Huginn and Ouro under paired edits | Requires external data, models and recorded lab layout |
| relevance_controls | Compare relevant and irrelevant edits; Huginn initialization schedules | Requires diagnostic data, audit outputs and models |
| rbs_node_readout | Adapt the RBS graph-node embedding readout | Requires upstream graph dataset and checkpoint |
| rbs_order_controls | Candidate swap and edge permutation on a frozen cohort | Requires the parent node-readout run |
| common_backbone_feasibility | DIRECT, EXPAND and REFINE execution, learning, cost and interventions | Data generator included; local GPT-2 and GPU required |

Scripts in experiments preserve the uploaded source bytes. Older exploratory
scripts and backup copies are excluded; see docs/source_inventory.json.

## Quick checks without a GPU
From the repository root:
```bash
python scripts/verify_artifacts.py
python scripts/audit_benchmarks.py
python experiments/rbs_node_readout/run_rbs_node_readout.py --self-test
```
These checks do not execute model training.

## RQ2 smoke run on an allocated GPU
Activate the recorded coconut_gpu environment first.
```bash
python scripts/run_rq2.py --model-dir /absolute/path/to/gpt2_model -- --mode smoke
```
The launcher writes run artifacts under ignored local/rq2_runs.
For the larger recorded configuration:
```bash
python scripts/run_rq2.py --model-dir /absolute/path/to/gpt2_model -- --mode tiny --train-groups 256 --dev-groups 128 --updates 1600 --batch-size 4 --lr 5e-5 --seed 2026 --eval-every 200 --save-weights
```
This is a configuration from the supplied source and reported lab run.
The GPU run has not been re-executed as part of this packaging review.

## Reproduction guide
Read docs/reproduction.md for original-checkpoint and RBS commands.
Read docs/review_findings.md before publication.
The repository now includes the original two-chain datasets, relevance-control
data and token audits, run summaries, selected analysis tables and the supplied
SIM-CoT wrapper. results/run_index.csv indexes the checkpoint trace summaries.
Model weights and raw traces remain external. Some combined analysis scripts
and final plotting inputs are still missing; see docs/review_findings.md.

## Recorded RQ2 pilot
The following values are read from the supplied saved summary, not rerun here.
Development comprises 256 questions from 128 graphs and one training seed.

| Strategy | Correct questions | Accuracy | Both queries correct |
| --- | --- | --- | --- |
| DIRECT | 224/256 | 87.50% | 101/128 |
| EXPAND | 224/256 | 87.50% | 97/128 |
| REFINE | 209/256 | 81.64% | 83/128 |

DIRECT is the lower-cost reference. This pilot does not establish hybrid gains.
The exact run configuration is configs/rq2_feasibility/recorded_pilot.json.

## Research status
The pilots support state-extraction and implementation feasibility.
Fixed-hybrid benefits and causal graph-information use remain research questions.
See docs/completed_and_planned_work.md and docs/limitations.md.

## Large files
Keep checkpoints and raw traces on the lab server or a separately managed
artifact service. Commit source, configs, compact tables and selected figures.
Use checkpoint revisions and SHA256 checksums to identify external artifacts.
Run the file verifier before committing. No GitHub upload has been performed.

## Attribution and licence
See THIRD_PARTY_NOTICES.md and LICENSE. Confirm the licence for your research
code before a public release. Upstream licences remain attached to their files.
