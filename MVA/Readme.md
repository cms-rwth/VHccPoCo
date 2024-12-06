Note: PyTorch is needed for GNN trainings. It is not needed for LGBM and DNN, but it is imported into the script so run:

## Setup 
```
pip install torch       #if not already done
pip install alive_progress
pip install optuna
```
or use the conda equivalent (not tested).

## Example usage for LGBM and DNN:

```
python training_lgbm_dnn.py train ../../output_VHcc_v01/Saved_columnar_arrays_ZLL/ --signal ZH_Hto2C_Zto2L_2022_preEE --background DYto2L-2Jets_MLL-50_FxFx_2022_preEE --model_type dnn
```

## Example usage for GNN:

```
python training.py -d ../../output_VHcc_v26_2L_full2022/ ../../output_VHcc_v25_1L_full2022/ ../../output_VHcc_v24_0L_full2022/
```
Point to multiple channels and eras. The GNN is conditioned on the channel and era.

For hyperparameter optimization:
```
python training.py -d ../../output_VHcc_v26_2L_full2022/ ../../output_VHcc_v25_1L_full2022/ ../../output_VHcc_v24_0L_full2022/ --hyperparameter --ntrials 20
```
Supports multi-GPU nodes.

For reloading/evaluating an old training, e.g., after hyperparamter optimization:

```
python training.py -d ../../output_VHcc_v26_2L_full2022/ ../../output_VHcc_v25_1L_full2022/ ../../output_VHcc_v24_0L_full2022/ --loadmodel /path/to/saved/model.pt
```


DNN and GNN trainings are faster on GPUs.
