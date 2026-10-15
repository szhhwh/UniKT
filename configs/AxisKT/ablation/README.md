# AxisKT difficulty ablations

The two model flags default to `true` and can be set under `model` in any
AxisKT training YAML:

| Variant | `use_rasch_embedding` | `use_irt_head` |
|---|---|---|
| Full model | true | true |
| Without embedding difficulty | false | true |
| Without IRT head | true | false |
| Without both | false | false |

Disabling Rasch embeddings substitutes zero difficulty in event embeddings,
local KC writes, and fused inference. The IRT output still learns difficulty.
Disabling the IRT head uses the existing linear ability head directly as the
logit, instead of `a * (theta - beta)`. Embedding difficulty remains active.
Parameters are constructed identically for all variants; inactive parameters
remain present. Both flags enabled preserve the original model behavior.

The `ReKTP_*_best.yaml` files archive the original search configurations.
The runner converts the model name to AxisKT and removes the unsupported
`residual_scale` field, keeping the remaining model hyperparameters.

Run from the repository root using the project Python environment:

```sh
python scripts/run_axiskt_ablation.py --output runs/axiskt_ablation
python scripts/run_axiskt_ablation.py --output runs/axiskt_ablation --summarize
```

The first command generates and runs four variants over five folds on each
of the four datasets. It requires CUDA and prepared data under `data/`.
Each job has an isolated directory; completed jobs are skipped on restart.
Partial jobs require explicit recovery before restarting the queue.
For a single run, use `python train.py --config <yaml>` with the desired flags.

The runner uses seed 42 and archives the source configs and their hashes.
Validation AUC selects the checkpoint. Five folds share a fixed held-out test
set, so fold standard deviation measures model variation, not independent
test-set uncertainty. EdNet source settings do not request subsampling;
record the preprocessing and sampling settings when comparing results.

Verify the mechanisms with:

```sh
python -m pytest tests/unit/models/test_axiskt_difficulty_ablation.py -q
```

CUDA checks compare fused inference with the training path for all variants;
they are skipped when CUDA is unavailable.
