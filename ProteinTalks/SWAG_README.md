# SWAG training

ProteinTalks supports stochastic weight averaging-Gaussian (SWAG) during joint model training.

## Training phase

Training begins with AdamW and a `ReduceLROnPlateau` scheduler. The SWAG phase begins after the validation loss plateaus and the learning rate falls below the fraction of its initial value specified by `--swag_start_factor`. The learning rate is then set to `--swag_lr`. Model snapshots are collected according to `--swag_freq`, with at most `--swag_max_models` snapshots stored for covariance estimation.

## Command

```bash
python main.py \
    --dataset_file_dir /path/to/dataset/ \
    --trainval_file_prefix allcelltype_drugpair_crossdrug_ \
    --time_stamp_predict_drug 6_24_48 \
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

## Parameters

- `--swag_lr`: learning rate used during the SWAG phase
- `--swag_freq`: interval between snapshots in fixed-frequency collection
- `--swag_max_models`: maximum number of stored snapshots
- `--swag_start_factor`: learning-rate factor used by the start condition

## Outputs

Training saves `best_checkpoint.pt`. When SWAG has collected models, it also saves `swag_mean_checkpoint.pt`.
