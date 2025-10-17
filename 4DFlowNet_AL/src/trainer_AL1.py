from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController
from Network.SR4DFlowNet import SR4DFlowNet

def prepare_network(patch_size, res_increase, low_resblock, hi_resblock):
    # Prepare input
    input_shape = (patch_size, patch_size, patch_size, 1)
    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')

    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u, v, w, u_mag, v_mag, w_mag]

    # network & output
    net = SR4DFlowNet(res_increase)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, low_resblock, hi_resblock)
    model = tf.keras.Model(input_layer, prediction)

    return model

def load_indexes(index_file):
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode')
    return indexes

if __name__ == "__main__":
    QUICKSAVE = True
    data_dir = '../../data'
    #benchmark_file = f'{data_dir}/aortaTest_patches.csv'
    #training_file = f'{data_dir}/aortaTrain_patches.csv'
    #validate_file = f'{data_dir}/aortaVal_patches.csv'

    benchmark_file = f'{data_dir}/test_patches.csv'
    training_file = f'{data_dir}/train_patches.csv'
    validate_file = f'{data_dir}/val_patches.csv'

    # Training hyperparameters
    base_learning_rate = 4e-5
    epochs_train = 30
    epochs_retrain = 30
    batch_size = 20
    mask_threshold = 0.6
    rotation = 'discrete'
    patch_size = 12
    res_increase = 2
    low_resblock = 8
    hi_resblock = 4
    n_models = 5
    top_fraction = 0.10
    train_size_rate = 0.50
    random_state = 42
    uncertainty_metric = "var_max"  # "var_mean", "var_max"
    AL_loops = 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    folder_name = f'networks_trained_{timestamp}'
    
    comments = "AL"

    train_index = load_indexes(training_file)
    len_train_index = len(train_index)
    train_size = int(train_size_rate * len_train_index)
    train_indexes = train_index[:train_size, :]
    retrain_indexes = train_index[train_size:,:]
    val_indexes = load_indexes(validate_file)
    benchmark_indexes = load_indexes(benchmark_file)
    network_list = []

    for model_id in range(n_models):
        tf.random.set_seed(model_id * 100 + 7)
        np.random.seed(model_id * 100 + 13)

        initial_learning_rate = base_learning_rate
        variables = {
        "initial_learning_rate": initial_learning_rate,
        "epochs_wo_AL": epochs_train,
        "epochs_w_AL": epochs_retrain,
        "batch_size": batch_size,
        "mask_threshold": mask_threshold,
        "rotation": rotation,
        "patch_size": patch_size,
        "res_increase": res_increase,
        "low_resblock": low_resblock,
        "hi_resblock": hi_resblock,
        "model": f"{model_id}/{n_models}",
        "train_size_rate": train_size_rate,
        "top_fraction": top_fraction,
        "uncertainty_metric": uncertainty_metric,
        "comments": comments}

        # Reload datasets for each model (to reshuffle differently each time)
        ph_train = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        trainset = ph_train.initialize_dataset_preloaded(train_indexes, shuffle=True)

        ph_val = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        valset = ph_val.initialize_dataset_preloaded(val_indexes, shuffle=True)

        ph_bench = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        testset = ph_bench.initialize_dataset_preloaded(benchmark_indexes, shuffle=False)
        print("Datasets prepared.")

        # Model name → unique per ensemble member
        network_name = f'4DFlowNet_ensemble_{model_id+1}'
        network = TrainerController(
            patch_size, res_increase, initial_learning_rate, QUICKSAVE,
            network_name, low_resblock, hi_resblock
        )

        
        network.init_model_dir(folder_name)

        network.train_network(trainset, valset, n_epoch=epochs_train, testset=testset)
        network_list.append(network)

        folder = network.model_dir
        file_name = os.path.join(folder, "variables.txt")
        with open(file_name, "w") as f:
            for name, value in variables.items():
                f.write(f"{name} = {value}\n")


    for k in range(AL_loops):
        ph_retrain = PatchHandler3D(data_dir, patch_size, res_increase, 1, rotation, mask_threshold)
        retrainset = ph_retrain.initialize_dataset_preloaded(retrain_indexes, shuffle=False)

        nr_rows = ph_retrain.dataset_length
        results_max_val = []
        results_mean_val = []
        
        for i, data_pairs in enumerate(retrainset):
            u, v, w, u_hr, v_hr, w_hr, u_mag, v_mag, w_mag, venc, mask = data_pairs
            input_data = [u, v, w, u_mag, v_mag, w_mag]
            ensemble_preds = []

            for net in network_list:
                predictions = net.model(input_data, training=False)
                predictions = tf.squeeze(predictions, axis=0)
                ensemble_preds.append(predictions)
            
            ensemble_preds = tf.stack(ensemble_preds, axis=0)
            variance_volume = tf.math.reduce_variance(ensemble_preds, axis=0)

            # Compute statistics
            max_val = tf.reduce_max(variance_volume).numpy()
            mean_val = tf.reduce_mean(variance_volume).numpy()
            if n_models>1:
                print(f"Patch {i+1}/{nr_rows} - Max Variance: {max_val:.6f}, Mean Variance: {mean_val:.6f}")
            results_max_val.append(max_val)
            results_mean_val.append(mean_val)

        df = pd.DataFrame(retrain_indexes)
        df['var_mean'] = results_mean_val
        df['var_max'] = results_max_val

        if n_models > 1:
            threshold = df[uncertainty_metric].quantile(1 - top_fraction)
            print(threshold)
            df_most_uncertain = df[df[uncertainty_metric] >= threshold]
            df_remaining = df[df[uncertainty_metric] < threshold]
            df_most_uncertain.to_csv("most_uncertain.csv", index=False)
        elif n_models == 1:
            df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            n_top = int(len(df) * top_fraction)
            df_most_uncertain = df_shuffled.iloc[:n_top]
            df_remaining = df_shuffled.iloc[n_top:]

        cols_to_drop = ["var_mean", "var_max"]
        df_most_uncertain = df_most_uncertain.drop(columns=cols_to_drop, errors="ignore")
        df_remaining = df_remaining.drop(columns=cols_to_drop, errors="ignore")

        top_patches = df_most_uncertain.astype(str).to_numpy(dtype="unicode")
        retrain_indexes = df_remaining.astype(str).to_numpy(dtype="unicode")

        print("top_patches shape:", top_patches.shape, type(top_patches))
        print("remaining_patches shape:", retrain_indexes.shape)

        merged_array = np.concatenate((train_indexes, top_patches), axis=0)

        ph_train = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        trainset = ph_train.initialize_dataset_preloaded(merged_array, shuffle=True)
        
        for model in network_list:
            model.train_network(trainset, valset, n_epoch=epochs_retrain, testset=testset, retrain=True)