import tensorflow as tf
import numpy as np
import time
import os
from Network.SR4DFlowNet import SR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset


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


if __name__ == '__main__':
    # Data
    data_dir = '../data/aorta_CFD'
    filename = 'aorta03_LR.h5'
    output_dir = "../result"
    output_filename = 'aorta_result_DE_M5.h5'
    number_of_ensembles = 5

    # Ensemble model paths
    model_paths = [f"../models/4DFlowNet_ensemble_{i}/4DFlowNet_ensemble_{i}-best.h5" for i in range(1, number_of_ensembles + 1)]
    
    patch_size = 12
    res_increase = 2
    low_resblock = 8
    hi_resblock = 4

    models = []
    for path in model_paths:
        net = prepare_network(patch_size, res_increase, low_resblock, hi_resblock)
        net.load_weights(path)
        models.append(net)

    # Params
    patch_size = 12
    res_increase = 2
    batch_size = 20
    round_small_values = True

    # Network
    low_resblock = 8
    hi_resblock = 4

    # Setup dataset
    input_filepath = f"{data_dir}/{filename}"
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    # Dataset size
    nr_rows = dataset.get_dataset_len(input_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Loop rows
    for nrow in range(0, nr_rows):
        print("\n--------------------------")
        print(f"\nProcessed ({nrow+1}/{nr_rows}) - {time.ctime()}")

        # Load row
        dataset.load_vectorfield(input_filepath, nrow)
        print(f"Original image shape: {dataset.u.shape}")

        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        all_preds = []
        all_uncertainties = []

        start_time = time.time()
        for current_idx in range(0, data_size, batch_size):
            time_taken = time.time() - start_time
            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')

            patch_index = np.index_exp[current_idx:current_idx+batch_size]

            # Collect predictions from all ensemble members
            ensemble_outputs = []
            for net in models:
                sr_images = net.predict([
                    velocities[0][patch_index],
                    velocities[1][patch_index],
                    velocities[2][patch_index],
                    magnitudes[0][patch_index],
                    magnitudes[1][patch_index],
                    magnitudes[2][patch_index]
                ], verbose=0)
                ensemble_outputs.append(sr_images)

            # Shape (n_models, batch, X, Y, Z, 3)
            ensemble_outputs = np.stack(ensemble_outputs, axis=0)

            # Mean and variance
            mean_pred = np.mean(ensemble_outputs, axis=0)
            var_pred = np.var(ensemble_outputs, axis=0)

            all_preds.append(mean_pred)
            all_uncertainties.append(var_pred)

        # Stitch back
        all_preds = np.concatenate(all_preds, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)

        for i in range(3):  # u,v,w components
            # Mean prediction reconstruction
            v = pgen._patchup_with_overlap(all_preds[..., i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v = v * dataset.venc
            if round_small_values:
                v[np.abs(v) < dataset.velocity_per_px] = 0
            v = np.expand_dims(v, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}',
                                        dataset.velocity_colnames[i], v, compression='gzip')

            # Uncertainty reconstruction
            v_unc = pgen._patchup_with_overlap(all_uncertainties[..., i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v_unc = np.expand_dims(v_unc, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}',
                                        f"uncertainty_{dataset.velocity_colnames[i]}",
                                        v_unc, compression='gzip')

        # Save voxel spacing
        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}',
                                        dataset.dx_colname, new_spacing, compression='gzip')

    print("Done!")