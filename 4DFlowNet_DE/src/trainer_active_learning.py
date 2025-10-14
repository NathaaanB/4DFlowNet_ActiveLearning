import numpy as np
import tensorflow as tf
import os
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController

def load_indexes(index_file):
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode')
    return indexes

def compute_uncertainty(predictions):
    """
    Example uncertainty measure: predictive entropy
    predictions: np.ndarray, shape (N, num_classes)
    """
    eps = 1e-8
    entropy = -np.sum(predictions * np.log(predictions + eps), axis=1)
    return entropy

if __name__ == "__main__":
    QUICKSAVE = True
    data_dir = '../../data'

    body_data = 'aorta'
    benchmark_file = f'{data_dir}/{body_data}Test_patches.csv'
    training_file = f'{data_dir}/{body_data}Train_patches.csv'
    validate_file = f'{data_dir}/{body_data}Val_patches.csv'

    train_indexes = load_indexes(training_file)
    val_indexes = load_indexes(validate_file)
    benchmark_indexes = load_indexes(benchmark_file)

    # Hyperparameters
    base_learning_rate = 1e-5
    epochs = 80
    batch_size = 20
    mask_threshold = 0.6
    rotation = 'discrete'
    patch_size = 12
    res_increase = 2
    low_resblock = 8
    hi_resblock = 4
    n_models = 10
    n_active_loops = 3           
    initial_train_fraction = 0.1 
    query_fraction = 0.1         

    comments = f"Active Learning with {n_active_loops} loops, query fraction={query_fraction}"

    for model_id in range(n_models):
        print(f"\n===== Active Learning Ensemble Member {model_id+1}/{n_models} =====")

        tf.random.set_seed(model_id * 100 + 7)
        np.random.seed(model_id * 100 + 13)

        initial_learning_rate = (model_id+1)*base_learning_rate

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
            "model": f"{model_id}/{n_models}",
            "comments": comments}

        # Split training indexes into initial and unlabeled pools
        np.random.shuffle(train_indexes)
        n_initial = int(initial_train_fraction * len(train_indexes))
        labeled_pool = list(train_indexes[:n_initial])
        unlabeled_pool = list(train_indexes[n_initial:])

        ph_val = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        valset = ph_val.initialize_dataset_preloaded(val_indexes, shuffle=True)

        ph_bench = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        testset = ph_bench.initialize_dataset_preloaded(benchmark_indexes, shuffle=False)

        network_name = f'4DFlowNet_AL_{model_id+1}_{body_data}'
        network = TrainerController(
            patch_size, res_increase, initial_learning_rate, QUICKSAVE,
            network_name, low_resblock, hi_resblock
        )
        network.init_model_dir()

        for loop in range(n_active_loops):
            print(f"\n--- Active Learning Loop {loop+1}/{n_active_loops} ---")

            ph_train = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
            trainset = ph_train.initialize_dataset_preloaded(labeled_pool, shuffle=True)

            network.train_network(trainset, valset, n_epoch=epochs, testset=testset)

            if loop < n_active_loops - 1:
                # Compute uncertainty on unlabeled pool
                ph_unlabeled = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
                unlabeled_set = ph_unlabeled.initialize_dataset_preloaded(unlabeled_pool, shuffle=False)

                predictions = network()
                predictions = network.predict_dataset(unlabeled_set)
                uncertainty = compute_uncertainty(predictions)

                # Select most uncertain samples
                n_query = int(query_fraction * len(unlabeled_pool))
                query_indices = np.argsort(uncertainty)[-n_query:]

                # Move selected samples to labeled pool
                new_samples = [unlabeled_pool[i] for i in query_indices]
                labeled_pool.extend(new_samples)
                unlabeled_pool = [x for i, x in enumerate(unlabeled_pool) if i not in query_indices]

        folder = network.model_dir
        file_name = os.path.join(folder, "variables.txt")
        print(f"Saving variables to {file_name}...")

        with open(file_name, "w") as f:
            for name, value in variables.items():
                f.write(f"{name} = {value}\n")

    print("\n✅ Active Learning Ensemble training finished.")