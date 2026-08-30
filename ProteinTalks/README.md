# ProteinTalks

ProteinTalks is a neural ordinary differential equation model for jointly predicting perturbation-induced proteomic dynamics and drug efficacy or combination synergy.

## Model

The model takes the following inputs:

- a baseline proteomic profile;
- a protein-aligned perturbation descriptor;
- molecular feature vectors for the two drug inputs.

The neural ODE produces proteomic predictions corresponding to 6, 24 and 48 hours. A phenotype head combines the predicted proteomic trajectory with the drug features to produce a drug-efficacy or combination-synergy probability.

Protein-expression prediction uses mean squared error (MSE), and phenotype prediction uses binary cross-entropy (BCE). The joint objective is

`Loss = (1 - lambda_pheno) * MSE + lambda_pheno * BCE`,

with `lambda_pheno=0.8` by default.

## Installation

ProteinTalks uses Python 3.10.6. Install the Python dependencies from this directory:

```bash
pip install -r requirements.txt
```

## Input files

For a dataset prefix such as `allcelltype_drugpair_crossdrug_`, the data directory contains:

- `<prefix>node_Index.csv`
- `<prefix>expr.csv`
- `<prefix>drug_fp_phychem_A.csv`
- `<prefix>drug_fp_phychem_B.csv`
- `<prefix>loo_label.csv`
- `<prefix>pert.csv`
- `<prefix>pheno.csv`

`loo_label.csv` identifies the experiment and time point for each row. The model uses the baseline profile at 0 hours and predicts the profiles at 6, 24 and 48 hours.

## Training

Run training from the `ProteinTalks` directory:

```bash
python main.py \
    --dataset_file_dir /path/to/dataset/ \
    --trainval_file_prefix allcelltype_drugpair_crossdrug_ \
    --time_stamp_predict_drug 6_24_48 \
    --train_percent 0.7 \
    --val_percent 0.2 \
    --test_percent 0.1 \
    --batch_size 64 \
    --patience 500 \
    --learning_rate 0.0005 \
    --lambda_pheno 0.8 \
    --use_swag \
    --swag_lr 0.0005 \
    --swag_freq 1 \
    --swag_max_models 20 \
    --swag_start_factor 0.2
```

The default training path uses AdamW and `ReduceLROnPlateau`.

## Prediction

```bash
python main.py \
    --train_from_scratch predict \
    --cp_save_dir_best /path/to/model_checkpoint.pt \
    --dataset_file_dir /path/to/dataset/ \
    --test_file_prefix test_ \
    --hidden_size 64 \
    --dropout_rate 0 \
    --time_stamp_predict_drug 6_24_48
```

## Main files

- `main.py`: training and prediction entry point
- `dataset.py`: dataset loading and splitting
- `model.py`: ProteinTalks model (`ppODE`)
- `trainer.py`: joint training, validation and SWAG integration
- `swag.py`: SWAG parameter collection and sampling
- `metrics.py`: evaluation metrics
- `plot.py`: ROC and precision–recall plots
- `config.py`: command-line configuration
