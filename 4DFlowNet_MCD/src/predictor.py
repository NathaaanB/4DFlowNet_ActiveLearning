import os
import time
import numpy as np
import tensorflow as tf
from Network.PatchGenerator import PatchGenerator
from utils.ImageDataset import ImageDataset
from utils import prediction_utils
from Network.SR4DFlowNet import SR4DFlowNet, conv3d, resnet_block, upsample3d
import h5py

# ======================== Prepare network ========================
def prepare_network(patch_size, res_increase, low_resblock, hi_resblock):
    input_shape = (patch_size, patch_size, patch_size, 1)

    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')
    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u, v, w, u_mag, v_mag, w_mag]

    net = SR4DFlowNet(res_increase)
    prediction, logvar_pred = net.build_network(u, v, w, u_mag, v_mag, w_mag, low_resblock, hi_resblock)
    model = tf.keras.Model(input_layer, [prediction, logvar_pred])

    return model

# ======================== Main script ========================
dr = 5

if __name__ == '__main__':
    data_dir = '../../data'
    filename = 'aorta03_LR.h5'
    output_dir = "../result"
    output_filename = f'Dropout_aorta_dr{dr}_lr4e-5.h5'
    model_path = f"../models/4DFlowNet_dr{dr}_lr4e-5/4DFlowNet-best.h5"

    patch_size = 12
    res_increase = 2
    batch_size = 20
    low_resblock = 8
    hi_resblock = 4
    n_samples = 100 # MC Dropout passes
    round_small_values = True

    # ========== Load dataset ==========
    dataset = ImageDataset()
    pgen = PatchGenerator(patch_size, res_increase)
    input_filepath = os.path.join(data_dir, filename)
    nr_rows = dataset.get_dataset_len(input_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    # ========== Load network ==========
    print(f"Loading 4DFlowNet: {res_increase}x upsample")
    network = prepare_network(patch_size, res_increase, low_resblock, hi_resblock)
    network.load_weights(model_path)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # ========== Loop through dataset ==========
    for nrow in range(nr_rows):
        print("\n--------------------------")
        print(f"Processed ({nrow+1}/{nr_rows}) - {time.ctime()}")

        # Load vectorfield for current row
        dataset.load_vectorfield(input_filepath, nrow)
        print(f"Original image shape: {dataset.u.shape}")

        # Patchify
        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        # Initialize result arrays
        results_mean = np.zeros((0, patch_size*res_increase, patch_size*res_increase, patch_size*res_increase, 3))
        results_epi_unc = np.zeros_like(results_mean)
        results_all_preds = []

        start_time = time.time()

        # ===== Predict with MC Dropout =====
        for current_idx in range(0, data_size, batch_size):
            patch_index = np.index_exp[current_idx:current_idx+batch_size]

            batch_inputs = [
                velocities[0][patch_index],
                velocities[1][patch_index],
                velocities[2][patch_index],
                magnitudes[0][patch_index],
                magnitudes[1][patch_index],
                magnitudes[2][patch_index]]

            # MC Dropout forward passes
            mc_preds = []
            mc_var = []
            for _ in range(n_samples):
                sr_images, logvar_images = network(batch_inputs, training=True)
                mc_preds.append(sr_images.numpy())
                mc_var.append(np.exp(logvar_images.numpy())) # variance = exp(log(variance))

            mc_preds = np.stack(mc_preds, axis=0)
            results_all_preds.append(mc_preds)

            mean_pred = mc_preds.mean(axis=0)
            epi_unc = mc_preds.var(axis=0)

            mc_var = np.stack(mc_var, axis=0)
            aleatoric_unc = mc_var.mean(axis=0)

            # Append batch results
            results_mean = np.append(results_mean, mean_pred, axis=0)
            results_epi_unc = np.append(results_epi_unc, epi_unc, axis=0)
            results_aleatoric_unc = np.append(results_epi_unc, aleatoric_unc, axis=0)

            time_taken = time.time() - start_time
            print(f"\rProcessed {min(current_idx+batch_size, data_size)}/{data_size} patches - Elapsed: {time_taken:.2f}s", end='\r')

        print(f"\nAll patches processed - Total elapsed: {time.time()-start_time:.2f}s")

        # ========== Reconstruct full volume ==========
        for i in range(3):
            # Mean prediction
            v_mean = pgen._patchup_with_overlap(results_mean[..., i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v_mean = v_mean * dataset.venc
            if round_small_values:
                v_mean[np.abs(v_mean) < dataset.velocity_per_px] = 0
            v_mean = np.expand_dims(v_mean, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.velocity_colnames[i], v_mean, compression='gzip')

            # Epistemic uncertainty
            v_epi = pgen._patchup_with_overlap(results_epi_unc[..., i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v_epi = np.expand_dims(v_epi, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f"uncertainty_{dataset.velocity_colnames[i]}", v_epi, compression='gzip')

            #v_alea = pgen._patchup_with_overlap(results_aleatoric_unc[..., i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            #v_alea = np.expand_dims(v_epi, axis=0)
            #prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f"aleatoric_{dataset.velocity_colnames[i]}", v_alea, compression='gzip')

            gt_dir = "../../data/aorta03_HR.h5"
            with h5py.File(gt_dir, "r") as f_gt:
                gt = np.squeeze(f_gt[f'{dataset.velocity_colnames[i]}'][()])   # ground truth 3D volume
                pred = np.squeeze(v_mean)         # our current prediction for this component
                rel_err_voxel = np.abs(pred - gt)         
                prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f"error_{dataset.velocity_colnames[i]}", rel_err_voxel, compression='gzip')

        # Save new spacing if available
        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

    #results_all_preds = np.concatenate(results_all_preds, axis=1)  
    #shape: (n_samples, total_patches, px, px, px, 3)

    """for s in range(n_samples):
        for i in range(3):
            v_sample = pgen._patchup_with_overlap(results_all_preds[s, ..., i],
                                                pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v_sample = v_sample * dataset.venc
            v_sample = np.expand_dims(v_sample, axis=0)
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}',
                                        f"sample{s}_{dataset.velocity_colnames[i]}",
                                        v_sample, compression='gzip')"""

    print("\nPrediction completed!")