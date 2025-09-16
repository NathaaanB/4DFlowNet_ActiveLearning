import numpy as np
import h5py
from utils import evaluation_utils as e_utils
import pandas as pd

if __name__ == "__main__":

    #data_dir = "../data/michigan_cerebro_CFD/patient3-postOp"
    #hr_filename = "patient3-postOp_HR.h5"
    #lr_filename = 'patient3-postOp_LR_dv.h5'

    data_dir = "../data"
    hr_filename = "example_data_HR.h5"
    lr_filename = 'example_data.h5'

    prediction_dir = "../result"

    pred_names = [
    ]

    prediction_filename = "example_result.h5"
    metrics_name = "metrics_example_data_SR" 

    ground_truth_file = f"{data_dir}/{hr_filename}"
    prediction_files = [f"{prediction_dir}/{p}/{prediction_filename}" for p in pred_names]
    lr_file = f"{data_dir}/{lr_filename}"

    #peak_flow_idx = 30 
    peak_flow_idx = 0 

    with h5py.File(ground_truth_file, 'r') as hf:
        u_hr = np.asarray(hf['u'])
        v_hr = np.asarray(hf['v'])
        w_hr = np.asarray(hf['w'])

        T = len(hf.get("u"))

        #mask = np.asarray(hf['tof_mask'])
        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        nf_mask = 1.0 - mask
        boundary_mask, core_mask = e_utils._create_boundary_and_core_masks(mask, 0.1, 'voxels')

        X,Y,Z = mask.shape
        cov_a = np.sum(mask)/(X*Y*Z)
        cov_b = np.sum(boundary_mask)/(X*Y*Z)
        cov_c = np.sum(core_mask)/(X*Y*Z)
        ratio_b = np.sum(boundary_mask)/np.sum(mask)
        ratio_c = np.sum(core_mask)/np.sum(mask)

        print(' ')
        print(f'Coverage: {100*cov_a:.3f} %')
        #print(f'XYZ: {X*Y*Z}')
        #print(f'sum_b: {np.sum(boundary_mask)}')
        #print(f'sum_c: {np.sum(core_mask)}')
        #print(f'mask: {np.sum(mask)}')
        print(f'Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %')
        print(f'Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %')


    for i, pred_file in enumerate(prediction_files):
        name = pred_names[i]
        print(' ')
        print('-'*25)
        print(name)
        print('-'*25)

        with h5py.File(pred_file, mode = 'r') as pf:
            u_sr = np.asarray(pf['u'])
            v_sr = np.asarray(pf['v'])
            w_sr = np.asarray(pf['w'])

        tanh_rel_err = np.zeros((T,3))
        rel_err = np.zeros((T,3))
        abs_err = np.zeros((T,4))
        rmse = np.zeros((T,4))

        vnrmse = np.zeros((T,4))
        d_error = np.zeros((T,4))

        Ks = np.zeros((T,3,3))
        Ms = np.zeros((T,3,3))
        Rs = np.zeros((T,3,3))

        for t in range(T):
            rel_err[t,0] = (e_utils.calculate_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            rel_err[t,1] = (e_utils.calculate_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            rel_err[t,2] = (e_utils.calculate_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))

            tanh_rel_err[t,0] = (e_utils.calculate_tanh_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            tanh_rel_err[t,1] = (e_utils.calculate_tanh_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            tanh_rel_err[t,2] = (e_utils.calculate_tanh_relative_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))

            abs_err[t,0] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            abs_err[t,1] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            abs_err[t,2] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
            abs_err[t,3] = (e_utils.calculate_absolute_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

            rmse[t,0] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            rmse[t,1] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            rmse[t,2] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
            rmse[t,3] = (e_utils.calculate_rmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

            vnrmse[t,0] = (e_utils.calculate_vnrmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            vnrmse[t,1] = (e_utils.calculate_vnrmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            vnrmse[t,2] = (e_utils.calculate_vnrmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
            vnrmse[t,3] = (e_utils.calculate_vnrmse(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

            d_error[t,0] = (e_utils.calculate_directional_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], mask))
            d_error[t,1] = (e_utils.calculate_directional_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
            d_error[t,2] = (e_utils.calculate_directional_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
            d_error[t,3] = (e_utils.calculate_directional_error(u_sr[t], v_sr[t], w_sr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

            Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = e_utils.linreg(u_sr[t], u_hr[t], mask)
            Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = e_utils.linreg(v_sr[t], v_hr[t], mask)
            Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = e_utils.linreg(w_sr[t], w_hr[t], mask)

            Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = e_utils.linreg(u_sr[t], u_hr[t], boundary_mask)
            Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = e_utils.linreg(v_sr[t], v_hr[t], boundary_mask)
            Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = e_utils.linreg(w_sr[t], w_hr[t], boundary_mask)

            Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = e_utils.linreg(u_sr[t], u_hr[t], core_mask)
            Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = e_utils.linreg(v_sr[t], v_hr[t], core_mask)
            Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = e_utils.linreg(w_sr[t], w_hr[t], core_mask)
        
        print('HERE---------------------------')

        print('---------------------------')

        print('Total avg')
        rel_err_tot = np.mean(rel_err, axis=0)
        print(f'Relative error [Fluid] {rel_err_tot[0]:.1f}')
        print(f'Relative error [Bound] {rel_err_tot[1]:.1f}')
        print(f'Relative error [Core] {rel_err_tot[2]:.1f}')

        tanh_rel_err_tot = np.mean(tanh_rel_err, axis=0)
        print(f'tanh Relative error [Fluid] {tanh_rel_err_tot[0]:.1f}')
        print(f'tanh Relative error [Bound] {tanh_rel_err_tot[1]:.1f}')
        print(f'tanh Relative error [Core] {tanh_rel_err_tot[2]:.1f}')

        abs_err_tot = np.mean(abs_err, axis=0)
        print(f'Absolute error [Fluid] {abs_err_tot[0]:.3f}')
        print(f'Absolute error [Bound] {abs_err_tot[1]:.3f}')
        print(f'Absolute error [Core] {abs_err_tot[2]:.3f}')
        print(f'Absolute error [Non-F] {abs_err_tot[3]:.3f}')

        rmse_tot = np.mean(rmse, axis=0)
        print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.3f}')
        print(f'R.M.S.   error [Bound] {rmse_tot[1]:.3f}')
        print(f'R.M.S.   error [Core] {rmse_tot[2]:.3f}')
        print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.3f}')

        vnrmse_tot = np.mean(vnrmse, axis=0)
        print(f'vNRMSE error [Fluid] {vnrmse_tot[0]:.3f}')
        print(f'vNRMSE error [Bound] {vnrmse_tot[1]:.3f}')
        print(f'vNRMSE error [Core] {vnrmse_tot[2]:.3f}') 
        print(f'vNRMSE error [Non-F] {vnrmse_tot[3]:.3f}')

        d_error_tot = np.mean(d_error, axis=0)
        print(f'Directional error [Fluid] {d_error_tot[0]:.3f}')
        print(f'Directional error [Bound] {d_error_tot[1]:.3f}')
        print(f'Directional error [Core] {d_error_tot[2]:.3f}')
        print(f'Directional error [Non-F] {d_error_tot[3]:.3f}')


        Rs_tot = np.mean(Rs, axis=0)
        print(f'U R2     [Fluid] {Rs_tot[0][0]:.3f}')
        print(f'U R2     [Bound] {Rs_tot[0][1]:.3f}')
        print(f'U R2     [Core] {Rs_tot[0][2]:.3f}')

        print(f'V R2     [Fluid] {Rs_tot[1][0]:.3f}')
        print(f'V R2     [Bound] {Rs_tot[1][1]:.3f}')
        print(f'V R2     [Core] {Rs_tot[1][2]:.3f}')

        print(f'W R2     [Fluid] {Rs_tot[2][0]:.3f}')
        print(f'W R2     [Bound] {Rs_tot[2][1]:.3f}')
        print(f'W R2     [Core] {Rs_tot[2][2]:.3f}')
        Ks_tot = np.mean(Ks, axis=0)

        print(f'U   K   [Fluid] {Ks_tot[0][0]:.3f}')
        print(f'U   K   [Bound] {Ks_tot[0][1]:.3f}')
        print(f'U   K   [Core] {Ks_tot[0][2]:.3f}')

        print(f'V   K   [Fluid] {Ks_tot[1][0]:.3f}')
        print(f'V   K   [Bound] {Ks_tot[1][1]:.3f}')
        print(f'V   K   [Core] {Ks_tot[1][2]:.3f}')

        print(f'W   K   [Fluid] {Ks_tot[2][0]:.3f}')
        print(f'W   K   [Bound] {Ks_tot[2][1]:.3f}')
        print(f'W   K   [Core] {Ks_tot[2][2]:.3f}')

        print('-  '*9)
        print('Peak Flow')

        print(f'Relative error [Fluid] {rel_err[peak_flow_idx][0]:.1f}')
        print(f'Relative error [Bound] {rel_err[peak_flow_idx][1]:.1f}')
        print(f'Relative error [Core] {rel_err[peak_flow_idx][2]:.1f}')

        print(f'tanh Relative error [Fluid] {tanh_rel_err[peak_flow_idx][0]:.1f}')
        print(f'tanh Relative error [Bound] {tanh_rel_err[peak_flow_idx][1]:.1f}')
        print(f'tanh Relative error [Core] {tanh_rel_err[peak_flow_idx][2]:.1f}')

        print(f'Absolute error [Fluid] {abs_err[peak_flow_idx][0]:.3f}')
        print(f'Absolute error [Bound] {abs_err[peak_flow_idx][1]:.3f}')
        print(f'Absolute error [Core] {abs_err[peak_flow_idx][2]:.3f}')
        print(f'Absolute error [Non-F] {abs_err[peak_flow_idx][3]:.3f}')

        print(f'R.M.S.   error [Fluid] {rmse[peak_flow_idx][0]:.3f}')
        print(f'R.M.S.   error [Bound] {rmse[peak_flow_idx][1]:.3f}')
        print(f'R.M.S.   error [Core] {rmse[peak_flow_idx][2]:.3f}')
        print(f'R.M.S.   error [Non-F] {rmse[peak_flow_idx][3]:.3f}')

        print(f'vNRMSE   error [Fluid] {vnrmse[peak_flow_idx][0]:.3f}')
        print(f'vNRMSE   error [Bound] {vnrmse[peak_flow_idx][1]:.3f}')
        print(f'vNRMSE   error [Core] {vnrmse[peak_flow_idx][2]:.3f}')
        print(f'vNRMSE   error [Non-F] {vnrmse[peak_flow_idx][3]:.3f}')

        print(f'Directional   error [Fluid] {d_error[peak_flow_idx][0]:.3f}')
        print(f'Directional   error [Bound] {d_error[peak_flow_idx][1]:.3f}')
        print(f'Directional   error [Core] {d_error[peak_flow_idx][2]:.3f}')
        print(f'Directional   error [Non-F] {d_error[peak_flow_idx][3]:.3f}')

        print(' ')
        print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]:.4f} \t m: {Ms[peak_flow_idx][0][0]:.4f} \t r^2: {Rs[peak_flow_idx][0][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]:.4f} \t m: {Ms[peak_flow_idx][0][1]:.4f} \t r^2: {Rs[peak_flow_idx][0][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][0][2]:.4f} \t m: {Ms[peak_flow_idx][0][2]:.4f} \t r^2: {Rs[peak_flow_idx][0][2]:.4f}')

        print(' ')
        print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]:.4f} \t m: {Ms[peak_flow_idx][1][0]:.4f} \t r^2: {Rs[peak_flow_idx][1][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]:.4f} \t m: {Ms[peak_flow_idx][1][1]:.4f} \t r^2: {Rs[peak_flow_idx][1][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][1][2]:.4f} \t m: {Ms[peak_flow_idx][1][2]:.4f} \t r^2: {Rs[peak_flow_idx][1][2]:.4f}')

        print(' ')
        print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]:.4f} \t m: {Ms[peak_flow_idx][2][0]:.4f} \t r^2: {Rs[peak_flow_idx][2][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]:.4f} \t m: {Ms[peak_flow_idx][2][1]:.4f} \t r^2: {Rs[peak_flow_idx][2][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][2][2]:.4f} \t m: {Ms[peak_flow_idx][2][2]:.4f} \t r^2: {Rs[peak_flow_idx][2][2]:.4f}')

        # Save metrics to csv
        metrics = {
            'lr_filename': lr_filename,
            'sr_filename': prediction_filename,

            'Coverage [%]': 100*cov_a,
            'Fluid Coverage [%]': 100*cov_b,
            'Core Coverage [%]': 100*cov_c,
            'Ratio Boundary/Core [%]': 100*ratio_c,

            'Relative error [Fluid]': rel_err_tot[0],
            'Relative error [Bound]': rel_err_tot[1],
            'Relative error [Core]': rel_err_tot[2],
            'tanh Relative error [Fluid]': tanh_rel_err_tot[0],
            'tanh Relative error [Bound]': tanh_rel_err_tot[1],
            'tanh Relative error [Core]': tanh_rel_err_tot[2],
            'Absolute error [Fluid]': abs_err_tot[0],
            'Absolute error [Bound]': abs_err_tot[1],
            'Absolute error [Core]': abs_err_tot[2],
            'Absolute error [Non-F]': abs_err_tot[3],
            'R.M.S. error [Fluid]': rmse_tot[0],
            'R.M.S. error [Bound]': rmse_tot[1],
            'R.M.S. error [Core]': rmse_tot[2],
            'R.M.S. error [Non-F]': rmse_tot[3],
            'vNRMSE error [Fluid]': vnrmse_tot[0],
            'vNRMSE error [Bound]': vnrmse_tot[1],
            'vNRMSE error [Core]': vnrmse_tot[2],
            'vNRMSE error [Non-F]': vnrmse_tot[3],
            'Directional error [Fluid]': d_error_tot[0],
            'Directional error [Bound]': d_error_tot[1],
            'Directional error [Core]': d_error_tot[2],
            'Directional error [Non-F]': d_error_tot[3],
            'U R2     [Fluid]': Rs_tot[0][0],
            'U R2     [Bound]': Rs_tot[0][1],
            'U R2     [Core]': Rs_tot[0][2],
            'V R2     [Fluid]': Rs_tot[1][0],
            'V R2     [Bound]': Rs_tot[1][1],
            'V R2     [Core]': Rs_tot[1][2],
            'W R2     [Fluid]': Rs_tot[2][0],
            'W R2     [Bound]': Rs_tot[2][1],
            'W R2     [Core]': Rs_tot[2][2],

            'U K     [Fluid]': Ks_tot[0][0],
            'U K     [Bound]': Ks_tot[0][1],
            'U K     [Core]': Ks_tot[0][2],
            'V K     [Fluid]': Ks_tot[1][0],
            'V K     [Bound]': Ks_tot[1][1],
            'V K     [Core]': Ks_tot[1][2],
            'W K     [Fluid]': Ks_tot[2][0],
            'W K     [Bound]': Ks_tot[2][1],
            'W K     [Core]': Ks_tot[2][2],


            'PEAK FLOW INDEX:': peak_flow_idx,
            'Relative error [Fluid] Peak': rel_err[peak_flow_idx][0],
            'Relative error [Bound] Peak': rel_err[peak_flow_idx][1],
            'Relative error [Core] Peak': rel_err[peak_flow_idx][2],
            'tanh Relative error [Fluid] Peak': tanh_rel_err[peak_flow_idx][0],
            'tanh Relative error [Bound] Peak': tanh_rel_err[peak_flow_idx][1],
            'tanh Relative error [Core] Peak': tanh_rel_err[peak_flow_idx][2],
            'Absolute error [Fluid] Peak': abs_err[peak_flow_idx][0],
            'Absolute error [Bound] Peak': abs_err[peak_flow_idx][1],
            'Absolute error [Core] Peak': abs_err[peak_flow_idx][2],
            'Absolute error [Non-F] Peak': abs_err[peak_flow_idx][3],
            'R.M.S. error [Fluid] Peak': rmse[peak_flow_idx][0],
            'R.M.S. error [Bound] Peak': rmse[peak_flow_idx][1],
            'R.M.S. error [Core] Peak': rmse[peak_flow_idx][2],
            'R.M.S. error [Non-F] Peak': rmse[peak_flow_idx][3],
            'vNRMSE error [Fluid] Peak': vnrmse[peak_flow_idx][0],
            'vNRMSE error [Bound] Peak': vnrmse[peak_flow_idx][1],
            'vNRMSE error [Core] Peak': vnrmse[peak_flow_idx][2],
            'vNRMSE error [Non-F] Peak': vnrmse[peak_flow_idx][3],
            'Directional error [Fluid] Peak': d_error[peak_flow_idx][0],
            'Directional error [Bound] Peak': d_error[peak_flow_idx][1],
            'Directional error [Core] Peak': d_error[peak_flow_idx][2],
            'Directional error [Non-F] Peak': d_error[peak_flow_idx][3],

            'U [Fluid] k': Ks[peak_flow_idx][0][0],
            'U [Bound] k': Ks[peak_flow_idx][0][1],
            'U [Core] k': Ks[peak_flow_idx][0][2],
            'U [Fluid] m': Ms[peak_flow_idx][0][0],
            'U [Bound] m': Ms[peak_flow_idx][0][1],
            'U [Core] m': Ms[peak_flow_idx][0][2],
            'U [Fluid] r^2': Rs[peak_flow_idx][0][0],
            'U [Bound] r^2': Rs[peak_flow_idx][0][1],
            'U [Core] r^2': Rs[peak_flow_idx][0][2],

            'V [Fluid] k': Ks[peak_flow_idx][1][0],
            'V [Bound] k': Ks[peak_flow_idx][1][1],
            'V [Core] k': Ks[peak_flow_idx][1][2],
            'V [Fluid] m': Ms[peak_flow_idx][1][0],
            'V [Bound] m': Ms[peak_flow_idx][1][1],
            'V [Core] m': Ms[peak_flow_idx][1][2],
            'V [Fluid] r^2': Rs[peak_flow_idx][1][0],
            'V [Bound] r^2': Rs[peak_flow_idx][1][1],
            'V [Core] r^2': Rs[peak_flow_idx][1][2],

            'W [Fluid] k': Ks[peak_flow_idx][2][0],
            'W [Bound] k': Ks[peak_flow_idx][2][1],
            'W [Core] k': Ks[peak_flow_idx][2][2],
            'W [Fluid] m': Ms[peak_flow_idx][2][0],
            'W [Bound] m': Ms[peak_flow_idx][2][1],
            'W [Core] m': Ms[peak_flow_idx][2][2],
            'W [Fluid] r^2': Rs[peak_flow_idx][2][0],
            'W [Bound] r^2': Rs[peak_flow_idx][2][1],
            'W [Core] r^2': Rs[peak_flow_idx][2][2],

        }

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_filename = pred_file.replace(prediction_filename, f'{metrics_name}.csv')

        metrics_df.to_csv(metrics_filename, index=False)
