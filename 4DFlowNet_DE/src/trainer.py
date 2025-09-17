import numpy as np
import tensorflow as tf
import os
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController

def load_indexes(index_file):
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode')
    return indexes

if __name__ == "__main__":
    QUICKSAVE = True
    data_dir = '../../data'
    benchmark_file = f'{data_dir}/aortaTest_patches.csv'
    training_file = f'{data_dir}/aortaTrain_patches.csv'
    validate_file = f'{data_dir}/aortaVal_patches.csv'

    train_indexes = load_indexes(training_file)
    val_indexes = load_indexes(validate_file)
    benchmark_indexes = load_indexes(benchmark_file)

    # Training hyperparameters
    initial_learning_rate = 2e-4
    epochs = 80
    batch_size = 20
    mask_threshold = 0.6
    rotation = 'discrete'
    patch_size = 12
    res_increase = 2
    low_resblock = 8
    hi_resblock = 4
    n_models = 10 

    variables = {
        "initial_learning_rate": initial_learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "mask_threshold": mask_threshold,
        "rotation": rotation,
        "patch_size": patch_size,
        "res_increase": res_increase,
        "low_resblock": low_resblock,
        "hi_resblock": hi_resblock,
        "n_models": n_models
    }

    for model_id in range(n_models):
        print(f"\n===== Training ensemble member {model_id+1}/{n_models} =====")

        # Different random seeds for diversity
        tf.random.set_seed(model_id * 100 + 7)
        np.random.seed(model_id * 100 + 13)

        # Reload datasets for each model (to reshuffle differently each time)
        ph_train = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        trainset = ph_train.initialize_dataset_preloaded(train_indexes, shuffle=True)

        ph_val = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        valset = ph_val.initialize_dataset_preloaded(val_indexes, shuffle=True)

        ph_bench = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        testset = ph_bench.initialize_dataset_preloaded(benchmark_indexes, shuffle=False)

        # Model name → unique per ensemble member
        network_name = f'4DFlowNet_ensemble_{model_id+1}'
        network = TrainerController(
            patch_size, res_increase, initial_learning_rate, QUICKSAVE,
            network_name, low_resblock, hi_resblock
        )
        network.init_model_dir()

        # Train this ensemble member
        network.train_network(trainset, valset, n_epoch=epochs, testset=testset)

    print("\n✅ Ensemble training finished. Models saved in ../models/")

    folder = network.model_dir
    file_name = os.path.join(folder, "variables.txt")

    print(f"Saving variables to {file_name}...")

    with open(file_name, "w") as f:
        for name, value in variables.items():
            f.write(f"{name} = {value}\n")
