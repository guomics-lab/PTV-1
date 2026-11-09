# PTNC_evaluation_64compounds

This file is part of the former ProteinTalks materials and is under the MIT License. The license text is `src/proteintalks/LICENCE.md`. That license covers only `src/proteintalks/` and `docs/proteintalks/`. The directory tree below describes a separately downloaded evaluation package, not this repository.

This package provides a more detailed analysis of the PTNC evaluation reported in the main text, including a comparison between the original 81-compound evaluation and the 64-compound evaluation. It contains the input data, trained model checkpoints, prediction code and saved outputs for evaluating ProteinTalks, GeneCompass, Geneformer and UCE on the 64-compound subset comprising 133 drug–cell line pairs. The supplied scripts perform inference with existing checkpoints; they do not retrain the models or regenerate the baseline embeddings.

## Supplementary note: compound identification across libraries in published studies

Different compound libraries can contain the same compounds under different source-specific identifiers. Published resources address chemical identity and sample provenance through two complementary practices: mapping source-specific records to standardized chemical identities and retaining original library or sample identifiers for traceability. These practices can be used together. The following examples illustrate the retention of original identifiers or source records:

1. Chandrasekaran et al., Nature Methods (2024) , with corresponding authors Shantanu Singh and Anne E. Carpenter at the Broad Institute, reported two identifiers each for BVT-948, dexamethasone and thiostrepton.
2. Brungs et al., Nature Methods (2025), with corresponding authors Robin Schmid (UCSD) and Tomáš Pluskal (IOCB Prague), retained different sample identifiers for entries sharing the same standardized chemical structure in Supplementary File 1.
3. Stokes et al., Cell (2020), with corresponding authors Regina Barzilay and James J. Collins at MIT, retained the original screening data for all 2,560 entries, including duplicate compounds, in Supplementary Table S1A.
 
These examples provide context for preserving source records while distinguishing them from chemical identity. For the 64-compound PTNC evaluation, eligibility was determined by compound identity rather than library-specific perturbation identifiers.

## Dataset provenance and evaluation scope

PTDS and PTNC were generated in separate experiments using compound libraries purchased from different suppliers. Library-specific perturbation identifiers were retained to trace each entry to its source library and experimental records. These identifiers describe experimental provenance and do not, by themselves, indicate distinct chemical identities.

After accounting for differences in compound naming and salt forms, we identified 17 compounds and 40 drug–cell line pairs shared between PTDS and PTNC. To assess performance specifically on compounds not represented in PTDS, we repeated the evaluation using the remaining 64 compounds and 133 drug–cell line pairs. Results from the 81-compound evaluation reported in the original paper and the 64-compound evaluation are compared below.

## Original 81-compound and 64-compound evaluation

The tables below summarize the 81-compound evaluation reported in the original paper and the 64-compound evaluation.

### Table 1. Mean performance metrics (81-compound evaluation → 64-compound evaluation)

| Model | AUPRC | AUROC | Accuracy |
| --- | --- | --- | --- |
| ProteinTalks | 0.937 → 0.946 | 0.890 → 0.879 | 0.845 → 0.839 |
| GeneCompass | 0.717 → 0.764 | 0.528 → 0.543 | 0.420 → 0.358 |
| Geneformer | 0.700 → 0.741 | 0.512 → 0.509 | 0.306 → 0.263 |
| UCE | 0.695 → 0.737 | 0.498 → 0.495 | 0.306 → 0.263 |

### Table 2. Updated statistical comparisons (p-values)

| Comparison | AUPRC | AUROC | Accuracy |
| --- | ---: | ---: | ---: |
| ProteinTalks vs. GeneCompass | 3.14 × 10⁻⁴ | 5.14 × 10⁻⁴ | 1.31 × 10⁻³ |
| ProteinTalks vs. Geneformer | 3.02 × 10⁻⁹ | 4.04 × 10⁻⁹ | 7.20 × 10⁻⁸ |
| ProteinTalks vs. UCE | 1.16 × 10⁻⁷ | 8.25 × 10⁻⁷ | 7.20 × 10⁻⁸ |

The overall findings remain consistent with those reported in the original paper, and the 64-compound evaluation does not alter the study’s conclusions.


The data and code supporting this analysis are available on https://drive.google.com/drive/folders/1XDATO9U_VM7C3Jg2IyeqL0hytKlP1ntB?usp=sharing.

## Directory structure

The root directory name may differ after downloading or extracting the package. Repeated output directories and long generated task names are abbreviated below.

```text
tableS7_reproduce_64CID 3/
├── README.md
├── run_ptv_predict.sh
├── run_scfm_predict.sh
├── code/
│   ├── ptv/                         # ProteinTalks model and inference code
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── predict.py
│   │   ├── metrics.py
│   │   ├── swag.py
│   │   ├── multi_task_learning.py
│   │   ├── stratify_celline.py
│   │   ├── plot.py
│   │   ├── plot_result.py
│   │   ├── utils.py
│   │   └── __init__.py
│   └── scfm/
│       └── temporal_expr_drug_predict_scFMs.py
├── checkpoints/
│   ├── ptv/                         # Five ProteinTalks checkpoints
│   └── scfm/                        # Fifteen downstream-model checkpoints
├── data/
│   └── 64CID/
│       ├── kept_cids_64.txt
│       └── allcelltype_drugpair_crossdrug_test12091214_*.csv
├── predictions/
│   ├── ptv_{run}/                   # Five run directories
│   │   └── prediction_{task}/
│   │       ├── predictions.csv
│   │       ├── predictions.npz
│   │       └── predictions_proteomics.npz
│   └── scfm_{model}_seed{seed}/      # Fifteen model/seed directories
│       ├── test_predict.npy
│       └── test_true.npy
└── logs/
    ├── ptv_{run}.log
    └── scfm_{model}_seed{seed}.log
```

## Input data

All input files are in `data/64CID/`. The CSV prefix is `allcelltype_drugpair_crossdrug_test12091214_`.

| Filename suffix or file | Contents |
| --- | --- |
| `kept_cids_64.txt` | CID list for the 64-compound evaluation subset |
| `loo_label.csv` | Experiment identifiers and time points used to align input records |
| `node_Index.csv` | Protein identifiers corresponding to the feature ordering |
| `expr.csv` | Proteomic expression input |
| `pert.csv` | Perturbation descriptors |
| `pheno.csv` | Drug-efficacy labels |
| `drug_fp_phychem_A.csv` | Molecular fingerprints and physicochemical features for drug input A |
| `drug_fp_phychem_B.csv` | Molecular fingerprints and physicochemical features for drug input B |
| `exp0_genecompass_embs.csv` | Precomputed baseline embeddings for GeneCompass |
| `exp0_geneformer_embs.csv` | Precomputed baseline embeddings for Geneformer |
| `exp0_uce_embs.csv` | Precomputed baseline embeddings for UCE |

The package documentation records 133 drug–cell line pairs: HS578T, 41; BT549, 37; MDA-MB-468, 28; and HCC1806, 27.

Baseline records are designated by `no` in the perturbation label and time point 0 in the model inputs. The supplied baseline embeddings are retained for inference and are not regenerated by the launch scripts. Preserve the supplied row ordering and file formats when using the loaders.

This package contains the prepared 64-compound subset. A full PTDS–PTNC structure-matching table, the detailed salt-standardization procedure and the original 81-compound input dataset are not included in the current folder.

## Models and checkpoints

### ProteinTalks

Five checkpoints are supplied in `checkpoints/ptv/`:

| Run | Checkpoint filename |
| --- | --- |
| `swa1` | `swa1_162_best_checkpoint.pth` |
| `swa2A` | `swa2A_162_best_checkpoint.pth` |
| `swa2B` | `swa2B_162_best_checkpoint.pth` |
| `runA` | `runA_seed2026_197_best_checkpoint.pth` |
| `runB` | `runB_seed1337_123_best_checkpoint.pth` |

### GeneCompass, Geneformer and UCE

The `checkpoints/scfm/` directory contains five downstream-model checkpoints per model, with seeds `1995`, `1999`, `327`, `508` and `531`. These models use the supplied baseline embeddings and drug-related inputs. The package does not contain the full foundation-model weights used to generate the embeddings.

Checkpoint filenames follow `{model}_seed{seed}_{epoch}best_checkpoint.pth`:

| Model | Seed 1995 | Seed 1999 | Seed 327 | Seed 508 | Seed 531 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeneCompass | 1000 | 800 | 800 | 800 | 800 |
| Geneformer | 800 | 1200 | 800 | 800 | 1200 |
| UCE | 1400 | 800 | 800 | 800 | 1000 |

The numbers in this table are the epoch labels in the supplied checkpoint filenames.

## Running inference

### Environment and paths

Use a Python environment with PyTorch and the dependencies imported by the code, including NumPy, pandas, scikit-learn, SciPy, Matplotlib, torcheval and torchdyn. The current folder does not include a version-pinned environment file. The launchers were prepared for a Bash/Linux environment with CUDA; `run_scfm_predict.sh` uses associative arrays and requires Bash 4 or later.

Both launchers currently contain the fixed path:

```bash
ROOT=/share/ppode_moa/tableS7_reproduce_64CID
```

Before running them, change `ROOT` in both files to the absolute path of this package on your machine. The supplied scripts contain unquoted path expansions, so use a package path **without spaces** (for example, rename the downloaded folder to `tableS7_reproduce_64CID`). Create the output directories if needed, and preserve a copy of the supplied predictions and logs if you want to compare them with a new run.

From the package root:

```bash
mkdir -p logs predictions
bash run_ptv_predict.sh 0
bash run_scfm_predict.sh 0
```

The optional argument is the CUDA device index; it defaults to `0`.

### ProteinTalks outputs

`run_ptv_predict.sh` loads the five checkpoints and runs inference on `data/64CID/` using `--train_from_scratch predict` and the 6/24/48-hour prediction setting. Each run writes to a generated subdirectory under `predictions/ptv_{run}/`.

- `predictions.csv`: ground-truth labels, predicted probabilities, binary predictions and experiment identifiers.
- `predictions.npz`: saved prediction arrays.
- `predictions_proteomics.npz`: saved proteomic prediction arrays.
- `logs/ptv_{run}.log`: execution log.

### Baseline-model outputs

`run_scfm_predict.sh` runs each model/seed combination with `--predict_only`, loads the corresponding checkpoint and saves:

- `test_predict.npy`: predicted scores.
- `test_true.npy`: corresponding ground-truth labels.
- `logs/scfm_{model}_seed{seed}.log`: execution log.
