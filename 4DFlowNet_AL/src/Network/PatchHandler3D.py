import tensorflow as tf
import numpy as np
import h5py
from scipy.ndimage import affine_transform
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class PatchHandler3D():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, rotation='discrete', mask_threshold=0.5):
        
        self.patch_size = patch_size
        
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold
        self.rotation = rotation # 'discrete' or 'affine'

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.mag_colnames  = ['mag_u', 'mag_v', 'mag_w']
        self.venc_colnames = ['venc_u', 'venc_v', 'venc_w']
        self.mask_colname = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def initialize_dataset_preloaded(self, indexes, shuffle=True):
        print("Preloading all patches into memory...")
        #preloaded_data = self.preload_patches(indexes)
        preloaded_data = self.preload_patches_parallell(indexes, max_workers=128)
        self.dataset_length = len(indexes)
        if shuffle:
            np.random.shuffle(preloaded_data)

        components = list(zip(*preloaded_data))  # List of 9 elements, each a list of tensors 
        components = [np.stack(arr, axis=0) for arr in components]  # Make each a proper array

        dataset = tf.data.Dataset.from_tensor_slices(tuple(components))

        dataset = dataset.batch(self.batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def preload_patches(self, indexes):
        """
        Pre-load all patches into memory and return a list of tensors.
        """
        all_data = []
        N_i = len(indexes)
        for i, idx_row in enumerate(indexes):
            if i % 100 == 0:
                print("Loading patch:", idx_row [:3], i, "of", N_i)
            data = self.load_patches_from_index(idx_row)
            all_data.append(data)
        return all_data

    def preload_patches_parallell(self, indexes, max_workers=64):
        """
        Pre-load all patches into memory in parallel using threading, with a tqdm progress bar.
        """
        load_func = self.load_patches_from_index

        # Wrap the function to support tqdm's executor.map tracking
        def wrapper(idx_row):
            return load_func(idx_row)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_data = list(tqdm(executor.map(wrapper, indexes), total=len(indexes), desc="Preloading patches"))

        return all_data

    def load_patches_from_index(self, index_row):
        """
        Eager version of the original load_patches_from_index_file.
        Expects one row of index file (list/array of 9 elements).
        """
        lr_hd5path = os.path.join(self.data_directory, index_row[0])
        hd5path    = os.path.join(self.data_directory, index_row[1])

        idx = int(index_row[2])
        x_start, y_start, z_start = map(int, index_row[3:6])
        is_rotate = int(index_row[6])
        rotation_plane = int(index_row[7])
        rotation_degree_idx = int(index_row[8])

        patch_size = self.patch_size
        hr_patch_size = patch_size * self.res_increase

        patch_index = np.index_exp[idx, x_start:x_start+patch_size, y_start:y_start+patch_size, z_start:z_start+patch_size]
        hr_patch_index = np.index_exp[idx, x_start*self.res_increase:x_start*self.res_increase +hr_patch_size, y_start*self.res_increase:y_start*self.res_increase+hr_patch_size, z_start*self.res_increase:z_start*self.res_increase +hr_patch_size]
        mask_index = np.index_exp[0, x_start*self.res_increase:x_start*self.res_increase+hr_patch_size, y_start*self.res_increase:y_start*self.res_increase+hr_patch_size, z_start*self.res_increase:z_start*self.res_increase+hr_patch_size]

        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index)

        if is_rotate > 0:

            if self.rotation == 'discrete':
                u_patch, v_patch, w_patch = self.apply_rotation_discrete(u_patch, v_patch, w_patch, rotation_degree_idx, rotation_plane, True)
                u_hr_patch, v_hr_patch, w_hr_patch = self.apply_rotation_discrete(u_hr_patch, v_hr_patch, w_hr_patch, rotation_degree_idx, rotation_plane, True)
                mag_u_patch, mag_v_patch, mag_w_patch = self.apply_rotation_discrete(mag_u_patch, mag_v_patch, mag_w_patch, rotation_degree_idx, rotation_plane, True)
                #mag_patch = self.rotate_object_discrete(mag_patch, rotation_degree_idx, rotation_plane)
                mask_patch = self.rotate_object_discrete(mask_patch, rotation_degree_idx, rotation_plane)

            elif self.rotation == 'affine':
                #print(f"Rotation matrix (plane={rotation_plane}, k={rotation_degree_idx}):\n", get_rotation_matrix(rotation_plane, rotation_degree_idx))
                u_patch, v_patch, w_patch = self.apply_rotation_affine(u_patch, v_patch, w_patch, rotation_degree_idx, rotation_plane)
                u_hr_patch, v_hr_patch, w_hr_patch = self.apply_rotation_affine(u_hr_patch, v_hr_patch, w_hr_patch, rotation_degree_idx, rotation_plane)
                mag_u_patch, mag_v_patch, mag_w_patch = self.apply_rotation_affine(mag_u_patch, mag_v_patch, mag_w_patch, rotation_degree_idx, rotation_plane, True)
                #mag_patch = self.rotate_object_affine(mag_patch, rotation_degree_idx, rotation_plane)
                mask_patch = self.rotate_object_affine(mask_patch, rotation_degree_idx, rotation_plane)

            else:
                raise ValueError(f"Unknown rotation type: {self.rotation}")

        return (
            u_patch[..., np.newaxis].astype(np.float32),
            v_patch[..., np.newaxis].astype(np.float32),
            w_patch[..., np.newaxis].astype(np.float32),
            u_hr_patch[..., np.newaxis].astype(np.float32),
            v_hr_patch[..., np.newaxis].astype(np.float32),
            w_hr_patch[..., np.newaxis].astype(np.float32),
            mag_u_patch[..., np.newaxis].astype(np.float32),
            mag_v_patch[..., np.newaxis].astype(np.float32),
            mag_w_patch[..., np.newaxis].astype(np.float32),
            np.array([venc], dtype=np.float32),
            mask_patch.astype(np.float32))             

    def apply_rotation_affine(self, u, v, w, rotation_idx, plane_nr):
        R = get_rotation_matrix(plane_nr, rotation_idx)
        u_rot = rotate_volume(u, R, order=0)
        v_rot = rotate_volume(v, R, order=0)
        w_rot = rotate_volume(w, R, order=0)
        u, v, w = apply_rigid_rotation(u_rot, v_rot, w_rot, R)
        return u, v, w

    def rotate_object_affine(self, img, rotation_idx, plane_nr, order=0):
        R = get_rotation_matrix(plane_nr, rotation_idx)
        rotated = rotate_volume(img, R, order)
        #print("Before/after rotation - mean:", img.mean(), rotated.mean())
        return rotated
    
    def apply_rotation_discrete(self, u, v, w, rotation_idx, plane_nr, is_phase_image):
        if rotation_idx == 1:
            # print("90 degrees, plane", plane_nr)
            u,v,w = rotate90(u,v,w, plane_nr, rotation_idx, is_phase_image)
        elif rotation_idx == 2:
            # print("180 degrees, plane", plane_nr)
            u,v,w = rotate180_3d(u,v,w, plane_nr, is_phase_image)
        elif rotation_idx == 3:
            # print("270 degrees, plane", plane_nr)
            u,v,w = rotate90(u,v,w, plane_nr, rotation_idx, is_phase_image)

        return u, v, w

    def rotate_object_discrete(self, img, rotation_idx, plane_nr):
        if plane_nr==1:
            ax = (0,1)
        elif plane_nr==2:
            ax = (0,2)
        elif plane_nr==3:
            ax = (1,2)
        else:
            # Unspecified rotation plane, return original
            return img

        img = np.rot90(img, k=rotation_idx, axes=ax)
        return img

    def _normalize(self, u, venc):
        return u / venc

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file

            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                mask = hl.get(self.mask_colname)[mask_index[1:]]
            mask = (mask >= self.mask_threshold) * 1.
            
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = hl.get(self.venc_colnames[i])[idx]

                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)

        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1

        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')

    def load_vectorfield_owo(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index):
        '''
        Load LR, HR and Magnitude components
        Also returns the venc and HR mask
        '''
        hires_images = []
        lowres_images = []

        # Load HR and mask
        with h5py.File(hd5path, 'r') as hl:
            for i in range(len(self.hr_colnames)):
                hires_images.append(hl[self.hr_colnames[i]][hr_patch_index])

            mask_key = next((k for k in self.mask_names if k in hl), None)
            if mask_key is None:
                raise KeyError(f"Unexpected mask name")
            
            try:
                mask = hl[mask_key][mask_index]
            except:
                mask = hl[mask_key][mask_index[1:]]
            mask = (mask >= self.mask_threshold)
            
        # Load LR, MAG, and VENC
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                lowres_images.append(hl[self.lr_colnames[i]][patch_index])

            mag_image = hl[self.mag_name][patch_index]
            mag_image = (mag_image / 4095.) # Normalize magnitude

            #venc = np.float32(hl[self.venc_name][idx])

            # Fallback to 'high_venc' if venc_name not found
            venc_key = self.venc_name if self.venc_name in hl else 'high_venc'
            venc = hl[venc_key][()]
            venc = np.squeeze(venc)

            # Handle different venc formats
            if np.isscalar(venc) or venc.ndim == 0:
                venc = np.float32(venc)
            elif venc.ndim == 1:
                venc = np.float32(venc[idx])
            else:
                raise ValueError(f"Unexpected venc")

        # Normalize velocity fields
        hires_images = self._normalize(np.asarray(hires_images), venc)
        lowres_images = self._normalize(np.asarray(lowres_images), venc)

        # U-LR, HR, V-LR, HR, W-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_image[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_image[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_image[2].astype('float32'),\
            venc.astype('float32'), mask.astype('float32')

def rotate180_3d(u, v, w, plane=1, is_phase_img=True):
    """
        Rotate 180 degrees to introduce negative values
        xyz Axis stays the same
    """
    if plane==1:
        # Rotate on XY, y*-1, z*-1
        ax = (0,1)
        if is_phase_img:
            v *= -1
            w *= -1
    elif plane==2:
        # Rotate on XZ, x*-1, z*-1
        ax = (0,2)
        if is_phase_img:
            u *= -1
            w *= -1
    elif plane==3:
        # Rotate on YZ, x*-1, y*-1
        ax = (1,2)
        if is_phase_img:
            u *= -1
            v *= -1
    else:
        # Unspecified rotation plane, return original
        return u,v,w
    
    # Do the 180 deg rotation
    u = np.rot90(u, k=2, axes=ax)
    v = np.rot90(v, k=2, axes=ax)
    w = np.rot90(w, k=2, axes=ax)    

    return u,v,w

def rotate90(u, v, w, plane, k, is_phase_img=True):
    """
        Rotate 90 (k=1) or 270 degrees (k=3)
        Introduce axes swapping and negative values
    """
    if plane==1:
        
        ax = (0,1)
        
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on XY, swap Z to Y +, Y to Z -
            temp = v
            v = w
            w = temp 
            if is_phase_img:
                w *= -1
        elif k == 3:
            # =================== ROTATION 270 =================== 
            # Rotate on XY, swap Z to Y -, Y to Z +
            temp = v
            v = w
            if is_phase_img:
                v *= -1
            w = temp  

    elif plane==2:
        ax = (0,2)
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on XZ, swap X to Z +, Z to X -
            temp = w
            w = u
            u = temp 
            if is_phase_img:
                u *= -1
        elif k == 3:
            # =================== ROTATION 270 =================== 
            # Rotate on XZ, swap X to Z -, Z to X +
            temp = w
            w = u
            if is_phase_img:
                w *= -1
            u = temp
        
    elif plane==3:
        ax = (1,2)
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on YZ, swap X to Y +, Y to X -
            temp = v
            v = u
            u = temp
            if is_phase_img:
                u *= -1
        elif k ==3:
            # =================== ROTATION 270 =================== 
            # Rotate on YZ, swap X to Y -, Y to X +
            temp = v
            v = u
            if is_phase_img:
                v *= -1
            u = temp
    else:
        # Unspecified rotation plane, return original
        return u,v,w
    
    # Do the 90 or 270 deg rotation
    u = np.rot90(u, k=k, axes=ax)
    v = np.rot90(v, k=k, axes=ax)
    w = np.rot90(w, k=k, axes=ax)    

    return u,v,w

def get_rotation_matrix(plane, k):
    theta = np.pi / 2 * k
    c, s = round(np.cos(theta), 1), round(np.sin(theta), 1)

    if plane == 1:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif plane == 2:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif plane == 3:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.eye(3)

def apply_rigid_rotation(u, v, w, R):
    velocity = np.stack([u, v, w], axis=-1)
    velocity_rot = np.einsum('...i,ji->...j', velocity, R)
    return velocity_rot[..., 0], velocity_rot[..., 1], velocity_rot[..., 2]

def rotate_volume(volume, R, order=0):
    """
    Rotate a 3D volume using an affine transformation and center the rotation.
    scipy.ndimage.affine_transform applies the inverse of the rotation matrix.
    """
    shape = volume.shape
    print("Volume shape:", shape)
    center = 0.5 * np.array(shape)
    
    # Compute inverse rotation matrix (affine_transform applies it)
    R_inv = np.linalg.inv(R)
    
    # Offset to ensure rotation is centered
    offset = center - R_inv @ center
    
    rotated = affine_transform(volume, R_inv, offset=offset, order=order, mode='constant', cval=0.0)
    return rotated
