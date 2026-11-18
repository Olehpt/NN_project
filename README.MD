scripts/prepare_dataset --- cleans up raw dataset for our use
scrips/data_processing --- process prepared dataset to .npz file, forms data to train, validate & test

python -m src.main train --model all --- train all 3 configurations of chronos 2

python -m src.main evaluate --model all --- run evaluation for all 3 configurations

python -m src.main forecast --model all --- run prediction on all 3 configurations

configs: zero-shot; full; lora.

