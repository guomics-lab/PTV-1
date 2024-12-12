# ppODE

This repository contains the code for training a neural network model to predict drug sensitivity and synergy in triple-negative breast cancer (TNBC) through proteomics data using the ppODE (Perturbation Proteomics Ordinary Differential Equation) architecture. The model is designed to dynamically analyze perturbed proteomic time series data.

## Usage

To run the script with the default parameters, use the following command:

```bash
python main.py --time_stamp_predict_drug "6_24_48" \
               --train_percent 0.7 \
               --val_percent 0.2 \
               --test_percent 0.1 \
               --taskname_prefix alltime_allpro_train_ratio_07train_02val \
               --trainval_file_prefix allcelltype_drugpair_crossdrug_ \
               --dataset_file_dir "./data/complete_data_proteo_structured_withcontrol20_0925_allproteins/" \
               --total_epoch 5000 \
               --patience 800
```

### Arguments

- **`--time_stamp_predict_drug`**: Time stamp used to predict drug synergy. Options include `"6"`, `"24"`, `"48"`, `"all"`, or `"6_24_48"`. Default: `"6"`.
  
- **`--lambda_pheno`**: Lambda parameter for multitask learning. Default: `0.8`.

- **`--taskname_prefix`**: Prefix for the task name and checkpoint files. Default: `""`.

- **`--dataset_file_dir`**: Directory for the dataset. Default: `"./data/complete_data_proteo_structured_withcontrol20_0925_allproteins/"`.

- **`--trainval_file_prefix`**: Prefix for the training and validation dataset files. Default: `"allcelltype_drugpair_"`.

- **`--test_file_prefix`**: If empty, the test set is determined by `--test_percent`; otherwise, the specified prefix is used for the test set. Default: `""`.

- **`--total_epoch`**: Total number of epochs for training. Default: `5000`.

- **`--patience`**: Number of epochs to wait before early stopping. Default: `500`.

- **`--train_percent`**: Percentage of data used for training. Default: `0.7`.

- **`--val_percent`**: Percentage of data used for validation. Default: `0.2`.

- **`--test_percent`**: Percentage of data used for testing. If this is set to `0.N`, a portion of the training/validation data will be used for testing. If `0`, the `--test_file_prefix` will define the test set. Default: `0.1`.

- **`--cp_save_dir_best`**: Directory to save the best checkpoint. If empty, the best checkpoint will be selected based on training performance.

- **`--batch_size`**: Batch size for training. Default: `128`.

### Example

To run the ppODE model with time stamps `"6_24_48"`, use the command:

```bash
python main.py --time_stamp_predict_drug "6_24_48" \
               --train_percent 0.7 \
               --val_percent 0.2 \
               --test_percent 0.1 \
               --taskname_prefix alltime_allpro_train_ratio_07train_02val \
               --trainval_file_prefix allcelltype_drugpair_crossdrug_ \
               --dataset_file_dir "./data/complete_data_proteo_structured_withcontrol20_0925_allproteins/" \
               --patience 800 \
               -- batch_size 64
```

## Dependencies

Before running the script, ensure that the following Python packages are installed:

- `torch`
- `torchdyn`
- `sklearn`
- `torcheval`

You can install these dependencies using pip:

```bash
pip install torch torchdyn sklearn torcheval
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
