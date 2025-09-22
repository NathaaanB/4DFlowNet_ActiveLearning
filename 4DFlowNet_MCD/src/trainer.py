import numpy as np
import os
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController

print("Code started")

def load_indexes(index_file):
    #Load patch index file (csv). This is the file that is used to load the patches based on x, y, z index
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    QUICKSAVE = True
    data_dir = '../../data'
    benchmark_file = '{}/aortaTest_patches.csv'.format(data_dir)
    training_file = '{}/aortaTrain_patches.csv'.format(data_dir)
    validate_file = '{}/aortaVal_patches.csv'.format(data_dir)

    print("Data loaded")

    restore = False
    if restore:
        model_dir = "../models/4DFlowNet_model_cd"
        model_file = "4DFlowNet-best.h5" 

    # Hyperparameters optimisation variables
    initial_learning_rate = 1e-5
    epochs =  100
    batch_size = 20
    dropout_rate = 0.20
    mask_threshold = 0.6
    rotation = 'discrete' #'discrete' or 'affine'
    comments = "Grid search"

     # Network setting
    network_name = '4DFlowNet'
    patch_size = 12
    res_increase = 2
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    low_resblock = 8
    hi_resblock = 4
    concrete_dropout = False

    variables = {
        "initial_learning_rate": initial_learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "mask_threshold": mask_threshold,
        "rotation": rotation,
        "patch_size": patch_size,
        "res_increase": res_increase,
        "low_resblock": low_resblock,
        "hi_resblock": hi_resblock,
        "comments": comments
    }

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
    trainset = z.initialize_dataset_preloaded(trainset, shuffle=True)

    # VALIDATION iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
    valset = valdh.initialize_dataset_preloaded(valset, shuffle=True)

    # # Benchmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        # WE use this benchmarking set so we can see the prediction progressing over time
        benchmark_set = load_indexes(benchmark_file)
        ph = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, rotation, mask_threshold)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset_preloaded(benchmark_set, shuffle=False)

    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, low_resblock, hi_resblock, dropout_rate, concrete_dropout)
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())

    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)

    folder = network.model_dir
    file_name = os.path.join(folder, "variables.txt")

    print(f"Saving variables to {file_name}...")

    with open(file_name, "w") as f:
        for name, value in variables.items():
            f.write(f"{name} = {value}\n")
