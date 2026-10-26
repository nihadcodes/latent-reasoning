# Experimental protocol
The two-chain pilot uses disjoint directed chains and two starting queries.
Four original conditions are original, relabeling, path change and answer change.
Within-model state comparisons retain all selected prompts without answer filtering.
Mirrored edits compare relevant-minus-irrelevant displacement within graph groups.
Resampling uses whole base groups to preserve pairing.

The RBS study uses a separate graph-trained symbolic checkpoint. An exact-k
walk frontier may overlap earlier frontiers. It is not always a shortest-distance shell.

The clean RQ2 pilot shares full GPT-2 initialization, data order and answer
supervision. DIRECT uses no extra update; EXPAND appends positions; REFINE
updates its one-slot workspace. Final readout still sees question tokens.
The diagnostic changes only final workspace states and preserves readout positions.
The source scripts define the precise conventions and checks.
