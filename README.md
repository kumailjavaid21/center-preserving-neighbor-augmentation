# Center-Preserving Neighbor Augmentation for Robust Multislice MRI Denoising

Research software and completed development measurements for training a multislice denoiser when neighboring observations may be displaced or have mismatched noise.

The generator receives four channels: observed center, frozen tile-wise noise estimate, previous-minus-center observation, and next-minus-center observation. It adds the predicted residual to the observed center. Paired training views retain the same center and acquired reference while perturbing neighbors through displacement, noise scaling, their combination, or context dropout. “Center-preserving” describes this input constraint; it does not guarantee preservation of anatomy or every output. Deployment uses the same generator and one native forward.

The three matched arms continue the same per-stream parent for 2,400 updates with fresh Adam states:

- **NOMINAL:** two nominal views.
- **AUGMENTED:** one nominal and one perturbed-neighbor view.
- **ANCHORED:** the augmented views plus the tested quality/context hinge penalties.

All three use the same samples, frozen noise estimator, model architecture and complete native-slice loss. The successful development comparison belongs to AUGMENTED versus NOMINAL. The additional anchor penalties did not improve the primary endpoint.

## Completed development evidence

| Projected comparison | PSNR difference | SSIM difference |
|---|---:|---:|
| AUGMENTED − NOMINAL, primary three-stress endpoint | +0.791062 dB | +0.039348 |
| AUGMENTED − NOMINAL, standard conditions | −0.040675 dB | −0.000636 |
| ANCHORED − AUGMENTED, primary endpoint | −0.001504 dB | See full tables |

The endpoint equally weights SHIFT1, SHIFT2 and NOISE_HIGH at position 7 and the four noisy conditions. Standard evaluation uses positions 0/4/7/11. Metrics are averaged within each subject, with two adaptation streams pooled within that same subject, then across **16 development subjects**. The streams are not 32 independent subjects and predictions are not ensembled. These subjects were repeatedly used during development; intervals are descriptive. AUGMENTED was selected after these results. The original standard retention targets were not met by AUGMENTED or ANCHORED in either stream. AUGMENTED also remains below CENTER under DIAGNOSTIC_SHIFT.

`results/` retains full-precision raw/projected, clean, subject, site, foreground, edge, adverse-tail, resource and ablation tables. It includes the actual contextual comparison set; Jiang is a local three-plane adaptation, and its BN-only diagnostic is separately labeled. No external published score is substituted. Original development bootstrap intervals are retained; no new interval construction replaces them. No independent confirmation result is included.

## Installation and scalar verification

Python 3.11 was used. The recorded training environment used PyTorch 2.5.1 with CUDA 12.1. Install an appropriate PyTorch build for your hardware and the pinned scientific dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/verify_results.py
python scripts/reproduce_tables.py --output reproduced_tables
```

Both result commands use the released CSV/JSON files and Python's standard library. They require no GPU, MRI or checkpoints. The first checks public file hashes and reconstructs the published subject-first estimates. The second writes the existing aggregate tables into a separate output directory. Source hashes and selection rules are recorded in `metadata/source_provenance.json`.

## Data and training prerequisites

Obtain the [IXI T1 dataset](https://brain-development.org/ixi-dataset/) under its applicable terms. No MRI, screenshots, prediction arrays, model weights, optimizer states or protected-cohort identifiers are distributed here. Acquired references contain intrinsic acquisition noise. Normalization uses the original nonzero 99.5th percentile of the canonical acquired reference volume; this is retrospective reference-derived preprocessing.

Provide the published TRAIN64 and development16 volumes from `metadata/development_population.json`. Create a private JSON manifest with a `subjects` list containing each `subject` and its T1 file `path`, relative to the manifest directory. The preparer accepts exactly that population; it has no archive scan or reserved-population loader.

```bash
python scripts/prepare_data.py --manifest private_data/subjects.json --output private_data/prepared
```

Exact continuation requires separately obtained trusted artifacts: each stream's frozen-parent FINAL14400, its CENTER FINAL1200 anchor, and the original frozen noise estimator. Their SHA256 identities are in `metadata/model_specifications.json`; their binaries are excluded. Random initialization does not reproduce these continuations. The preceding parent-training program and unrelated research history are outside this release.

```bash
python scripts/train.py --config configs/augmented.json --stream 20261117 --prepared-data private_data/prepared --parent-checkpoint private_data/parent_stream_a.pt --center-checkpoint private_data/center_stream_a.pt --noise-checkpoint private_data/frozen_noise.pt --anchor-cache runs/shared_anchors --output runs/augmented_stream_a
```

Use the matching config for each arm and repeat for stream 20261118 with its own parents. Common immutable scalar anchors are generated once per indexed training sample and reused across arms. Training preserves full native-slice gradients, two student views per case, five conditions per update, and the 128/64 tile/stride with ordered FP64 blending. It supports committed checkpoint resume and rejects ambiguous interrupted work. It does not choose an intermediate checkpoint from development scores. The extraction/import checks and scalar verification used for this release do not constitute a fresh end-to-end training replication.

To evaluate an authentic separately obtained final artifact on the same development population:

```bash
python scripts/evaluate.py --prepared-data private_data/prepared --checkpoint private_data/augmented_stream_a.pt --noise-checkpoint private_data/frozen_noise.pt --stream 20261117 --method AUGMENTED --output runs/development_cases.csv
```

This entry point supports NOMINAL, AUGMENTED, ANCHORED and the frozen NAF3_SCRATCH parent. It uses the original development corruption keys and evaluates the defined standard and stress cases. The contextual comparator measurements remain available for scalar verification; their earlier training workflows are not rerun by this release. The public adapters preserve the authenticated mathematical functions; exact final checkpoint reproduction still depends on the original data, parents, environment and numerical execution.

## Attribution and license notices

NAFNet: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang and Jian Sun, *Simple Baselines for Image Restoration*, ECCV 2022; [upstream repository](https://github.com/megvii-research/NAFNet), commit `2b4af71ebe098a92a75910c233a3965a3e93ede4`. The four-channel MRI stem, external residual readout and native renderer are local adaptations. The complete upstream NAFNet MIT and incorporated BasicSR Apache-2.0 notices are preserved in `THIRD_PARTY_LICENSES.txt`, including the original copyright notices. No additional blanket license is asserted for material whose licensing has not been established.

Jiang contextual architecture: Dongsheng Jiang et al., *Denoising of 3D magnetic resonance images with multi-channel residual learning of convolutional neural network*, Japanese Journal of Radiology (2018), [DOI](https://doi.org/10.1007/s11604-018-0758-8). The local three-plane implementation and training conditions are adaptations; no author software or weights are redistributed.

IXI acknowledges EPSRC grant GR/S21533/02 and records CC BY-SA 3.0 dataset terms. Consult the official dataset site before obtaining or redistributing its data. This repository distributes source and scalar research measurements, with no medical-image payloads.
