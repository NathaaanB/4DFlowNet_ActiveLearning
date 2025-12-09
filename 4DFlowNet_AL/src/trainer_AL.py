from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController
from Network.SR4DFlowNet import SR4DFlowNet
from metrics.most_uncertain_plots.plot_most_UQ_html import html_uncertain_patches

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

@tf.function
def process_patch(models, input_data):
    preds = [m.model(input_data, training=False) for m in models]
    preds = tf.stack(preds, axis=0)
    variance_volume = tf.math.reduce_variance(preds, axis=0)
    return tf.reduce_max(variance_volume), tf.reduce_mean(variance_volume)

def load_indexes(index_file):
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode')
    return indexes

if __name__ == "__main__":
    body_part = "aorta" # "cerebro" or "aorta" or "cardiac"
    QUICKSAVE = True
    data_dir = '../../data'
    benchmark_file = f'{data_dir}/{body_part}Test_patches.csv'
    training_file = f'{data_dir}/{body_part}Train_patches.csv'
    validate_file = f'{data_dir}/{body_part}Val_patches.csv'

    benchmark_file = f'{data_dir}/test_patches.csv'
    training_file = f'{data_dir}/train_patches.csv'
    validate_file = f'{data_dir}/val_patches.csv'

    # Training hyperparameters
    base_learning_rate = 4e-5
    epochs_train = 80
    n_models = 5
    top_fraction = 0.1
    train_size_rate = 0.36
    train_size = None
    random_state = 42
    uncertainty_metric = "var_mean"
    AL_loops = 25
    random_sampling = True

    loops_switch = 0

    batch_size = 20
    mask_threshold = 0.6
    rotation = 'discrete'
    patch_size = 12
    res_increase = 2
    low_resblock = 8
    hi_resblock = 4
    timestamp_folder = datetime.now().strftime("%Y%m%d-%H%M")
    comments = "AL"
    columns_names = ["source", "target", "index", "start_x", "start_y", "start_z", "rotate", "rotation_plane", "rotation_degree_idx", "coverage", "compartment"]


    train_index = load_indexes(training_file)
    len_train_index = len(train_index)
    if train_size == None:
        train_size = int(train_size_rate * len_train_index)

    np.random.seed(random_state)
    shuffled_indexes = np.random.permutation(train_index)

    train_indexes = shuffled_indexes[:train_size, :]
    retrain_indexes = shuffled_indexes[train_size:, :]
    #train_indexes = load_indexes(f"../models/networks_trained_AL_top1_train1/training_sets/training_set_AL{loops_switch+1}.csv") 
    #retrain_indexes = load_indexes(f"../models/networks_trained_AL_top1_train1/AL_loop_{loops_switch}/remaining_AL{loops_switch}_retraining.csv")
    val_indexes = load_indexes(validate_file)
    benchmark_indexes = load_indexes(benchmark_file)

    print(f"Initial train size: {train_indexes.shape}, retrain size: {retrain_indexes.shape}, val size: {val_indexes.shape}, benchmark size: {benchmark_indexes.shape}")

    network_list = []

    for k in range(loops_switch, AL_loops + loops_switch):
        folder_name = f'networks_trained_{timestamp_folder}/AL_loop_{k+1}'

        for model_id in range(n_models):
            tf.random.set_seed(model_id * 100 + 7)
            np.random.seed(model_id * 100 + 13)

            initial_learning_rate = base_learning_rate
            variables = {
            "initial_learning_rate": initial_learning_rate,
            "epochs_train": epochs_train,
            "batch_size": batch_size,
            "mask_threshold": mask_threshold,
            "rotation": rotation,
            "patch_size": patch_size,
            "res_increase": res_increase,
            "low_resblock": low_resblock,
            "hi_resblock": hi_resblock,
            "model": f"{model_id+1}/{n_models}",
            "train_size_rate": train_size_rate,
            "train_size": train_indexes.shape[0],
            "top_fraction": top_fraction,
            "uncertainty_metric": uncertainty_metric,
            "AL_loop": AL_loops,
            "training_file": training_file,
            "random_sampling": random_sampling,
            "comments": comments}

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
                network_name, low_resblock, hi_resblock)
            
            network.init_model_dir(folder_name)
            if model_id == 0:
                os.makedirs(f"../models/networks_trained_{timestamp_folder}/training_sets/", exist_ok=True)
                os.makedirs(f"../models/networks_trained_{timestamp_folder}/most_uncertain_sets/", exist_ok=True)
                os.makedirs(f"../models/networks_trained_{timestamp_folder}/remaining_sets/", exist_ok=True)
                os.makedirs(f"../models/networks_trained_{timestamp_folder}/retraining_sets/", exist_ok=True)
                df = pd.DataFrame(train_indexes, columns=columns_names)
                df.to_csv(f"../models/networks_trained_{timestamp_folder}/training_sets/training_set_AL{k+1}.csv", index=False)

            network.train_network(trainset, valset, n_epoch=epochs_train, testset=testset)
            network_list.append(network)

            folder = network.model_dir
            file_name = os.path.join(folder, "variables.txt")
            with open(file_name, "w") as f:
                for name, value in variables.items():
                    f.write(f"{name} = {value}\n")

        ph_retrain = PatchHandler3D(data_dir, patch_size, res_increase, 1, rotation, mask_threshold)
        retrainset = ph_retrain.initialize_dataset_preloaded(retrain_indexes, shuffle=False)

        results_max_val = []
        results_mean_val = []
        
        for i, data_pairs in enumerate(retrainset):
            u, v, w, u_hr, v_hr, w_hr, u_mag, v_mag, w_mag, venc, mask = data_pairs
            input_data = [u, v, w, u_mag, v_mag, w_mag]
            max_val, mean_val = process_patch(network_list, input_data)
         
            results_max_val.append(max_val)
            results_mean_val.append(mean_val)

        df = pd.DataFrame(retrain_indexes, columns=columns_names)
        results_max_val = tf.stack(results_max_val).numpy()
        results_mean_val = tf.stack(results_mean_val).numpy()
        df['var_mean'] = results_mean_val

        if (n_models > 1) and (not random_sampling):
            threshold = df[uncertainty_metric].quantile(1 - top_fraction)
            df_most_uncertain = df[df[uncertainty_metric] >= threshold]
            df_remaining = df[df[uncertainty_metric] < threshold]

            folder = f"../models/{folder_name}/"
            file_name = os.path.join(folder, "threshold.txt")
            with open(file_name, "w") as f:
                f.write(f"threshold_{k+1} = {threshold}\n")

        elif (n_models == 1) or random_sampling:
            df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            n_top = int(len(df) * top_fraction)
            df_most_uncertain = df_shuffled.iloc[:n_top]
            df_remaining = df_shuffled.iloc[n_top:]

        df_most_uncertain.to_csv(f"../models/networks_trained_{timestamp_folder}/most_uncertain_sets/most_uncertain_AL{k+1}.csv", index=False)
        df_remaining.to_csv(f"../models/{folder_name}/remaining_sets/remaining_patches_AL{k+1}.csv", index=False)

        cols_to_drop = ["var_mean"]
        df_most_uncertain = df_most_uncertain.drop(columns=cols_to_drop, errors="ignore")
        df_remaining = df_remaining.drop(columns=cols_to_drop, errors="ignore")
        df_remaining.to_csv(f"../models/{folder_name}/retraining_sets/remaining_AL{k+1}_retraining.csv", index=False)

        top_patches = df_most_uncertain.astype(str).to_numpy(dtype="unicode")
        retrain_indexes = df_remaining.astype(str).to_numpy(dtype="unicode")

        print("top_patches shape:", top_patches.shape, train_indexes.shape)
        print("remaining_patches shape:", retrain_indexes.shape)

        train_indexes = np.concatenate((train_indexes, top_patches), axis=0)
    
    df = pd.DataFrame(train_indexes, columns=columns_names)