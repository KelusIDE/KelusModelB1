# Kani Pixels

## Current model: Pixels v2

The current image experiment is a 18.96M-parameter, class-conditioned diffusion
transformer in `kelus/pixel_dit.py`. It finished 40,000 steps on 48,467 Open Images
photos across 174 subjects. It produces **64×64 pixels**; the larger preview tiles
are enlarged for display. The completed previews are still blurry and distorted.
It recognizes subject names, not arbitrary detailed scene descriptions.

Its weights were trained from random initialization. **V2 uses MLX** for automatic
differentiation and GPU execution. The original NumPy-only baseline is documented
below; the no-ML-framework description applies to that baseline, not to v2.

### Additional training

`python3 -m kelus.pixel_continue` copies the completed v2 checkpoint, including its
EMA weights and optimizer, into `training-runs/kani-pixels-v2-extended`. It queues
another **40,000 steps (80,000 total)** after existing GPU trainers finish. Batch
size stays 32, learning rate drops to 0.0001, and GPU memory is capped at 7 GiB.
An exclusive runner lock prevents duplicate continuation jobs. Do not start an
independent GPU training job while this continuation is running.

Progress, including the queued state:

```sh
python3 -m kelus.progress training-runs/kani-pixels-v2-extended.log --full
```

The runner's output is `training-runs/kani-pixels-v2-extended.queue.log`; training
output is `.out`, with checkpoints and status inside the run directory. Checkpoints
save every 1,000 steps and previews every 2,500. Fixed preview noise (seed 40000)
matches the preserved `baseline-040000.png`, so progress can be compared fairly.
New checkpoints also preserve the random-number generator state for resuming.
The original checkpoint predates that field, so its first continuation initializes
a new deterministic random stream while retaining trained weights and Adam state.

The existing model stays at `training-runs/kani-pixels-v2`; the continuation does
not automatically replace it. More steps do not increase native resolution or
guarantee good images. V2 currently reports training loss only, with no held-out
image evaluation; inspect the candidate before choosing it for the CLI/extension.

Verification (the resume test uses a tiny model on CPU):

```sh
.venv/bin/python -m unittest tests.test_pixel_dit tests.test_pixel_continue -v
```

### Sampling quality review

`kelus.pixel_review` compares Euler and Heun integration using the same initial
noise, subject order and model-evaluation budget. Euler remains the serving and
training-preview default. Heun is optional and evaluates the velocity field twice
per integration step; it does not change the weights or native resolution.

```sh
.venv/bin/python -m kelus.pixel_review \
  --run training-runs/kani-pixels-v2 \
  --out training-runs/pixel-quality-review/sampling --device cpu
```

The default review retains all 48 results: Cat, Dog, Car and Apple; seeds 2042 and
2043; guidance 1, 2 and 4; Euler with 60 steps versus Heun with 30. It writes each
native 64×64 image, a labeled native-size contact sheet, a nearest-neighbor 3×
display sheet, and a JSON report with the checkpoint SHA-256 and settings. It
refuses an existing output directory and rejects checkpoints that change during
the comparison. Repeat against a stable continuation checkpoint with a new
output folder to assess the next training stage. CPU is the review default so
the active GPU trainer can continue.

Single-image experimentation is also available:

```sh
.venv/bin/python -m kelus.pixel_dit sample \
  --run training-runs/kani-pixels-v2 --prompt 'a cat' --count 1 \
  --solver heun --steps 30 --guidance 2 --seed 2042 --device cpu \
  --scale 1 --columns 1 --out cat-native.png
```

This is a visual comparison, not a quality score or held-out evaluation. Numerical
solver accuracy does not guarantee recognizable subjects. The Heun test checks
against an analytic flow solution; another test proves that default Euler output
still exactly matches the original sampler. Second-order sampling is also used
in the [EDM reference implementation](https://github.com/NVlabs/edm/blob/main/generate.py);
our implementation integrates Kani's own rectified-flow velocity directly.

The first completed review at step 40,000 showed similar overall results between
the two solvers, with substantial subject and detail failures remaining. The
default was therefore kept unchanged. A separate fixed-noise label probe showed
that changing Cat/Dog/Car/Apple does affect the output, but the selected Cat and
Dog results remain difficult to distinguish. See the contact sheets and findings
in [`training-runs/pixel-quality-review/REVIEW.md`](training-runs/pixel-quality-review/REVIEW.md).

### Data audit (2026-10-23)

The 48,467 array entries match their source records and class indices, with no
exact duplicate pixel arrays. Class counts range from 41 to 300. All entries are
available to the optimizer: **there is no held-out split for v2**. Taking a new
subset from these previously used images would not create a clean evaluation.

The collector uses whole-image labels followed by a center crop. Inspection of
fixed samples found labeled subjects that occupy only a small part of the image
or are cut off; this can weaken the subject signal at 64×64. The contact sheet
and exact selected indices are in `training-runs/pixel-quality-review/` as
`training-examples.png` and `data-audit.json`. Object-centered crops and genuinely
unseen evaluation images are the next data improvements; this audit does not
establish that cropping is the only cause of poor generation.

For v2, the relevant dataset files are `images.npy`, `labels.npy`,
`manifest.json`, and `sources.jsonl`. The older `manifest.jsonl`, `images/`, and
`SOURCE_README.md` in the same directory belong to the separate Cooper Hewitt v1
experiment; they are not v2's image-caption training inputs.

## Original NumPy baseline (v1)

Kani Pixels replaces drawing-program generation as the direction for image research.
It predicts RGB pixels directly using text-conditioned rectified flow: training
interpolates between Gaussian noise and a real image, and our network learns the
velocity along that path. Sampling integrates predicted velocities from noise.

All weights start random. Network operations, gradients and AdamW use NumPy and
the repository's own autograd. No pretrained weights, external tokenizer, text
encoder, latent autoencoder, PyTorch, TensorFlow or diffusion framework is used.
Pillow only decodes/resizes/writes images.

## Current limits

This is a small MLP baseline at 32×32, not a high-resolution image generator.
Its hashed bag-of-words caption encoder does not understand word order. It has
no learned spatial hierarchy or attention. These are substantial limits; the
baseline establishes a measurable starting point, not an architecture promised
to scale to photorealism. Increasing canvas size or upscaling is not evidence of
new detail. A thousand well-captioned images of one domain can support an initial
experiment, not a claim of broad high-quality generation.

The existing Painter UI is not silently redirected to random/untrained weights.
Old Painter code/checkpoints remain available for rollback. A pixel checkpoint
must pass visual review and held-out prompt comparisons before serving it.

## Dataset

Supply a UTF-8 JSONL manifest with local image paths relative to the manifest:

```json
{"image":"images/001.jpg","caption":"a person wearing a blue coat","split":"train","source_url":"https://example.org/object/001","license":"CC0"}
{"image":"images/002.jpg","caption":"a person standing outdoors","split":"validation","source_url":"https://example.org/object/002","license":"CC0"}
```

Use actual captions describing visible content, not just filenames. Keep source
URLs and license information. Select a narrow domain, deduplicate before splitting,
and reserve approximately 10% for validation. Exact duplicate image files are
rejected across either split. Near-duplicates require a separate curation pass.
Validation images are not used for optimization. Images are center-cropped to a
square; inspect crops to avoid cutting off subjects. All decoded samples are
currently held in memory. The manifest checksum is recorded in each checkpoint.

## Train and sample

Install the existing NumPy requirements plus Pillow (`requirements-vision.txt`).
From the repository root:

```sh
python3 -m kelus.pixels train --data data/pixels/manifest.jsonl \
  --out checkpoints/kani-pixels-v1.npz --size 32 --width 256 \
  --steps 1000 --batch 8
python3 -m kelus.pixels sample --checkpoint checkpoints/kani-pixels-v1.npz \
  --prompt "a person wearing a blue coat" --out sample.png --seed 42
```

Training prints fixed-noise validation loss and saves atomically every 100 steps.
It refuses an already-existing output at startup. There is no resume yet; saved
checkpoints contain model parameters, not optimizer/RNG state. They use `.npz`,
not `.safetensors`. Changing serialization would not change training or quality.

Evaluate images at native resolution using unseen captions and several seeds.
Compare subject identity, composition, prompt changes, diversity, and nearest
training examples. Falling denoising/flow loss alone does not prove image quality
or exclude memorization. Do not deploy this baseline based only on its loss.

## Verification

```sh
python3 -m unittest tests.test_pixels -v
```

Tests cover actual optimization, prompt-conditioned pixels, deterministic sampling,
and checkpoint equivalence. The initial online-data experiment uses `data/make_pixels.py` to collect
1,000 unique CC0 Cooper Hewitt images with their accessibility descriptions
from `biglam/cooper-hewitt-alt-text`. It records source links and hashes, saves
64×64 center crops, then trains at 32×32. This is museum/design imagery, not
a balanced general-purpose image dataset. Collection and training logs live
in `training-runs/kani-pixels-v1/`. Captions may be long or use museum
terminology; visual and caption-quality review is still necessary.
