# Kani Big v2: free GPU continuation

The prepared experiment expands our own **88,118,784-parameter Kani** to
**173,090,304 parameters**, with 24 layers and a 2,048-token context. It starts
from the Big Kani v1 checkpoint at **step 35,500**. Added blocks initially
contribute zero to the residual stream, preserving the existing predictions.
Those blocks still need training; this is not evidence of GPT-4-level ability.

**Status on 2026-10-23:** the enlarged seed, upload bundle and notebook are
prepared locally. **Cloud training has not started.** Browser automation timed
out before a Colab runtime could be configured. The Mac's existing Big Kani v1
training continues independently.

## Start a free Colab session

1. Open [Google Colab](https://colab.research.google.com/) and choose
   **File → Upload notebook**. Select
   [`notebooks/kani-free-gpu.ipynb`](notebooks/kani-free-gpu.ipynb).
2. Choose **Runtime → Change runtime type → T4 GPU**. Use the free tier;
   no subscription or compute-unit purchase is needed for this setup.
3. Run the notebook cells in order. Review the Google Drive authorization
   prompt yourself. When the upload cell asks, select
   [`training-runs/cloud/kani-cloud.zip`](training-runs/cloud/kani-cloud.zip)
   (about 586 MB). The bundle contains our own checkpoint, source code and
   existing training corpus. It excludes credentials and unrelated projects.
4. Watch the training cell's progress bar. Checkpoints are copied to
   **My Drive → KaniTraining → kani-big-v2**. Allow a checkpoint save to finish
   before using **Runtime → Disconnect and delete runtime**.

Free GPUs and session lengths depend on availability; a free session is not
guaranteed. If Colab has no GPU quota available, try later rather than changing
to a paid runtime. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

The notebook requires a compatible NVIDIA driver and GPU, checks these before
training, and installs `mlx[cuda12]==0.32.4`. This follows the
[MLX CUDA requirements](https://ml-explore.github.io/mlx/build/html/install.html#cuda).
P100 GPUs are unsupported by this build. MLX supplies array operations and
automatic differentiation: this GPU path uses a framework, while preserving
our architecture, tokenizer and own weights. It downloads no pretrained model.

## Budget, saving and resuming

| Setting | Prepared value |
| --- | --- |
| Total continuation budget | 20,000 optimizer updates across sessions |
| Session budget | 3 hours, then save and stop |
| Batch | 1 sequence, accumulated over 4 microbatches |
| Context | 2,048 tokens |
| Learning rate | 0.0001 with warmup and decay |
| Checkpoint interval | 250 updates, plus clean interruption/session end |
| Memory settings | Gradient checkpointing and a 12 GiB MLX memory guideline |
| Persistent output | `MyDrive/KaniTraining/kani-big-v2/{latest,best}.npz` |

The full experiment presents 163,840,000 training tokens, sampled from the
existing approximately 164M-token corpus. Repeated samples are not new data.
Runtime speed and memory use on a real T4 have not yet been measured.

For another session, open the same notebook and run its cells again. It reuses
the Drive bundle and restores model weights, Adam state and random state.
Keep the total-step/batch/learning-rate recipe unchanged; the session-duration
limit may change. The trainer rejects a different corpus or resume recipe.
Only run one session against a given Drive backup folder at a time.

Interrupting the training cell requests a checkpoint after the current update.
A forced runtime termination can lose updates since the last Drive backup.
“Stopped” means the session ended before the target; “complete” means the update
budget finished. Neither is a capability evaluation. Raw pretraining still
needs instruction tuning and independent coding/reasoning tests before release.

## Local artifacts and checks

- [`training-runs/kani-big-v2-seed/seed.npz`](training-runs/kani-big-v2-seed/seed.npz):
  enlarged own-weight initialization, with no optimizer state.
- [`training-runs/kani-big-v2-seed/growth.json`](training-runs/kani-big-v2-seed/growth.json):
  parameter counts, parent hash and a measured initial logits difference of 0
  on the deterministic smoke-test sequence.
- [`training-runs/cloud/kani-cloud.json`](training-runs/cloud/kani-cloud.json):
  source checkpoint identity and per-file hashes; also included as `bundle.json`
  inside the upload archive. The notebook verifies every file before use.

Local tests pass for prediction-preserving growth, gradients reaching the added
blocks, checkpointed-gradient parity, and interrupted/resumed training matching
an uninterrupted run, including optimizer and random state. The notebook's
format and code-cell syntax were checked. CUDA execution and the rendered Colab
notebook have not been validated because no remote runtime was started.

To prepare a future bundle from a different completed checkpoint, use a new
filename (the builder refuses to overwrite an existing bundle):

```sh
.venv/bin/python -m kelus.cloud_bundle \
  --source training-runs/kani-big-v1/latest.npz \
  --out training-runs/cloud/kani-cloud-next.zip
```

That makes a new snapshot; it does not update an existing cloud run. Use a new
Drive experiment folder for a new parent or training recipe. The current seed
and bundle retain the step-35,500 parent even as the Mac trainer advances.

## Kani Advanced on Modal

When Kaggle's weekly GPU quota is used up, Kani Advanced continues on [Modal](https://modal.com) with
[`kaggle/modal_kani_advanced.py`](kaggle/modal_kani_advanced.py): the same training script on one L4 GPU,
about $9 a session out of Modal's free $30 a month. Its docstring lists the commands: upload Kaggle's
`checkpoint.pt` and `progress.json` and `modal deploy` it once, then `python kaggle/modal_kani_advanced.py start`
per session (it runs on Modal's side; this computer can sleep).
