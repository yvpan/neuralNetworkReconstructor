# for reproducible result
random_seed = 1 # global random seed
from numpy.random import seed
seed(random_seed)
import tensorflow as tf
tf.random.set_seed(random_seed)
import time
t0 = time.time()
print("Importing packages ...")
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
import os
import numpy as np
import scipy.stats as stats
from scipy.stats import lognorm, norm, sem
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
import sklearn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import pandas as pd 
import seaborn as sbn
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Input, MaxPooling3D, Conv3D, MaxPooling2D, Conv2D, Masking, Embedding, ELU, Add
from tensorflow.keras import backend as K
from radiotools import helper as hp
from NuRadioMC.utilities import medium
#from NuRadioMC.SignalProp import analyticraytracing as ray

def dropout(mat, rate):
    if rate == 1.:
        return np.zeros(mat.shape)
    elif rate == 0.:
        return mat
    elif rate > 1.:
        sys.exit("Dropout rate > 1")
    elif rate < 0.:
        sys.exit("Dropout rate < 0")
    mask = np.random.uniform(low = 0., high = 1., size = mat.shape) < 1. - rate
    return mask * mat / (1. - rate)

def normalTimeDiff(travel_times):
    tmp = np.zeros(travel_times.shape)
    for i in range(channelPerStr * strNum):
        for j in range(2):
            tmp += (travel_times - travel_times[:, i, j].reshape(travel_times.shape[0], 1, 1))
    tmp /= float(strNum * channelPerStr * 2.)
    return tmp

def my_mspe(y_true, y_pred):
    # self defined mean squared percentage error
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.square((y_true - y_pred) / (math_ops.abs(y_true) + K.epsilon()))
    return K.mean(diff, axis = -1)

def normalize(mat):
    # normalize a list of 2d arrays to be between 0 and 1
    nRow = mat.shape[1]
    nCol = mat.shape[2]
    mat = mat.reshape((mat.shape[0], nRow * nCol))
    maximum = np.nanmax(mat, axis = 1)
    maximum = maximum.reshape((maximum.shape[0], 1))
    minimum = np.nanmin(mat, axis = 1)
    minimum = minimum.reshape((minimum.shape[0], 1))
    deno = maximum - minimum + K.epsilon()# avoid dividing by zero
    mat = (mat - minimum) / deno
    return mat.reshape((mat.shape[0], nRow, nCol))

def periodic_padding_flexible(tensor, axis, padding = 1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)
    ndim = len(tensor.shape)
    for ax, p in zip(axis, padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left
        ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right, middle, left], axis = ax)
    return tensor

def upStream(layers, shareConv, inputs):
    if shareConv:
        # shared conv layers of different branches
        x = inputs
        paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT", constant_values = 0)
        x = periodic_padding_flexible(x, axis = 2, padding = 1)
        # groups > 1 only works on gpu
        x = Conv2D(nodes, kernel_size = (3, 3), padding = "valid", groups = 1, kernel_initializer = tf.keras.initializers.he_uniform(seed = random_seed))(x) # already manually padded
        x = ELU()(x)
        for i in range(layers - 1):
            paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
            x = tf.pad(x, paddings, "CONSTANT", constant_values = 0)
            x = periodic_padding_flexible(x, axis = 2, padding = 1)
            # groups > 1 only works on gpu
            x = Conv2D(nodes, kernel_size = (3, 3), padding = "valid", groups = 1, kernel_initializer = tf.keras.initializers.he_uniform(seed = random_seed))(x) # already manually padded
            x = ELU()(x)
            x = Dropout(rate = d1)(x)
        x = Flatten()(x)
        x = Dropout(rate = d2)(x)
    else:
        # separate conv layers of different branches
        x = inputs
    return x
    
def downStream(layers, shareConv, pred, inputs):
    if shareConv:
        # shared conv layers of different branches
        x = inputs
    else:
        # separate conv layers of different branches
        x = inputs
        paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT", constant_values = 0)
        x = periodic_padding_flexible(x, axis = 2, padding = 1)
        # groups > 1 only works on gpu
        x = Conv2D(nodes, kernel_size = (3, 3), padding = "valid", groups = 1, kernel_initializer = tf.keras.initializers.he_uniform(seed = random_seed))(x) # already manually padded
        x = ELU()(x)
        for i in range(layers - 1):
            paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
            x = tf.pad(x, paddings, "CONSTANT", constant_values = 0)
            x = periodic_padding_flexible(x, axis = 2, padding = 1)
            # groups > 1 only works on gpu
            x = Conv2D(nodes, kernel_size = (3, 3), padding = "valid", groups = 1, kernel_initializer = tf.keras.initializers.he_uniform(seed = random_seed))(x) # already manually padded
            x = ELU()(x)
            x = Dropout(rate = d1)(x)
        x = Flatten()(x)
        x = Dropout(rate = d2)(x)
    for i in range(layers):
        x = Dense(nodes, kernel_initializer = tf.keras.initializers.he_uniform(seed = random_seed))(x)
        x = ELU()(x)
        x = Dropout(rate = d3)(x)
    x = Dense(1, name = "{}_output".format(pred))(x)
    return x

def plotFeature(x, y):
    # plot relationships between different feature pairs
    print("Plotting features ...")
    data = {"t_dir[1, 0]": x[:, 1, 0, 0],
            "t_ref[1, 0]": x[:, 1, 0, 1],
            "a_dir[1, 0]": x[:, 1, 0, 2],
            "a_ref[1, 0]": x[:, 1, 0, 3],
            "r_dir[1, 0]": x[:, 1, 0, 4],
            "r_ref[1, 0]": x[:, 1, 0, 5],
            "f_dir[1, 0]": x[:, 1, 0, 6],
            "f_ref[1, 0]": x[:, 1, 0, 7],
            "v_dir[1, 0]": x[:, 1, 0, 8],
            "v_ref[1, 0]": x[:, 1, 0, 9],
            "c_dir[1, 0]": x[:, 1, 0, 10],
            "c_ref[1, 0]": x[:, 1, 0, 11],
            "t_dir[2, 0]": x[:, 2, 0, 0],
            "t_ref[2, 0]": x[:, 2, 0, 1],
            "a_dir[2, 0]": x[:, 2, 0, 2],
            "a_ref[2, 0]": x[:, 2, 0, 3],
            "r_dir[2, 0]": x[:, 2, 0, 4],
            "r_ref[2, 0]": x[:, 2, 0, 5],
            "f_dir[2, 0]": x[:, 2, 0, 6],
            "f_ref[2, 0]": x[:, 2, 0, 7],
            "v_dir[2, 0]": x[:, 2, 0, 8],
            "v_ref[2, 0]": x[:, 2, 0, 9],
            "c_dir[2, 0]": x[:, 2, 0, 10],
            "c_ref[2, 0]": x[:, 2, 0, 11],
            "t_dir[0, 1]": x[:, 0, 1, 0],
            "t_ref[0, 1]": x[:, 0, 1, 1],
            "a_dir[0, 1]": x[:, 0, 1, 2],
            "a_ref[0, 1]": x[:, 0, 1, 3],
            "r_dir[0, 1]": x[:, 0, 1, 4],
            "r_ref[0, 1]": x[:, 0, 1, 5],
            "f_dir[0, 1]": x[:, 0, 1, 6],
            "f_ref[0, 1]": x[:, 0, 1, 7],
            "v_dir[0, 1]": x[:, 0, 1, 8],
            "v_ref[0, 1]": x[:, 0, 1, 9],
            "c_dir[0, 1]": x[:, 0, 1, 10],
            "c_ref[0, 1]": x[:, 0, 1, 11],
            "rr": y[:, 0],
            "zz": y[:, 1],
            "pp": y[:, 3],
            "tt": y[:, 4],
            "az": y[:, 7],
            "ze": y[:, 8],
            "sh": y[:, 12]}
    df = pd.DataFrame(data, columns = ["t_dir[1, 0]", "t_ref[1, 0]", "a_dir[1, 0]", "a_ref[1, 0]", "r_dir[1, 0]", "r_ref[1, 0]", "f_dir[1, 0]", "f_ref[1, 0]", "v_dir[1, 0]", "v_ref[1, 0]", "c_dir[1, 0]", "c_ref[1, 0]", "t_dir[2, 0]", "t_ref[2, 0]", "a_dir[2, 0]", "a_ref[2, 0]", "r_dir[2, 0]", "r_ref[2, 0]", "f_dir[2, 0]", "f_ref[2, 0]", "v_dir[2, 0]", "v_ref[2, 0]", "c_dir[2, 0]", "c_ref[2, 0]", "t_dir[0, 1]", "t_ref[0, 1]", "a_dir[0, 1]", "a_ref[0, 1]", "r_dir[0, 1]", "r_ref[0, 1]", "f_dir[0, 1]", "f_ref[0, 1]", "v_dir[0, 1]", "v_ref[0, 1]", "c_dir[0, 1]", "c_ref[0, 1]", "rr", "zz", "pp", "tt", "az", "ze", "sh"])
    
    pd.plotting.scatter_matrix(df[:50], figsize = (60, 60), alpha = 0.6, diagonal = "hist")
    plt.savefig("{}/scat_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor))
    plt.clf()
    
    plt.subplots(figsize = (6.4, 4.8))
    corrMatrix = df.corr()
    sbn.heatmap(corrMatrix, vmin = -1., vmax = 1., cmap = cm.RdBu)
    plt.tight_layout()
    plt.savefig("{}/corr_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor))
    plt.clf()

def plotLossVsLr(history):
    lr = 1e-8 * (10 ** (np.arange(100) / 20))
    plt.plot(lr, np.array(history.history["loss"]), label = "training", linewidth = 0.5)
    plt.plot(lr, np.array(history.history["val_loss"]), label = "validation", linewidth = 0.5)
    plt.ylim((0., 1000.))
    plt.xlim((1e-8, 1e-3))
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("learning rate")
    plt.ylabel("total_loss")
    plt.tight_layout()
    plt.savefig("{}/lossVsLr_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor))
    plt.clf()

def plotLearn(pred, history):
    #plot learning curves
    if pred == "rr" or pred == "cos" or pred == "sin" or pred == "cosAz" or pred == "sinAz":
        plt.plot(np.sqrt(np.array(history.history["{}_output_loss".format(pred)])), label = "training", linewidth = 0.5)
        plt.plot(np.sqrt(np.array(history.history["val_{}_output_loss".format(pred)])), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 1.))
        plt.yscale('log')
    elif pred == "en" or pred == "sh":
        plt.plot(np.sqrt(np.array(history.history["{}_output_loss".format(pred)])), label = "training", linewidth = 0.5)
        plt.plot(np.sqrt(np.array(history.history["val_{}_output_loss".format(pred)])), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 10.))
        plt.yscale('log')
    elif pred == "tt" or pred == "ze":
        plt.plot(np.degrees(np.sqrt(np.array(history.history["{}_output_loss".format(pred)]))), label = "training", linewidth = 0.5)
        plt.plot(np.degrees(np.sqrt(np.array(history.history["val_{}_output_loss".format(pred)]))), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 100))
        plt.yscale('log')
    elif pred == "total":
        plt.plot(np.array(history.history["loss"]), label = "training", linewidth = 0.5)
        plt.plot(np.array(history.history["val_loss"]), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 1000))
        plt.yscale('log')
    elif pred == "combine":# combination of all the outputs with weights
        plt.plot(np.array(history.history["loss"]), label = "loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["rr_output_loss"]), label = "rr_output_loss", linewidth = 1)
        plt.plot(10000. * np.array(history.history["cos_output_loss"]), label = "cos_output_loss", linewidth = 1)
        plt.plot(10000. * np.array(history.history["sin_output_loss"]), label = "sin_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["cosAz_output_loss"]), label = "cosAz_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["sinAz_output_loss"]), label = "sinAz_output_loss", linewidth = 1)
        plt.plot(np.array(history.history["sh_output_loss"]), label = "sh_output_loss", linewidth = 1)
        plt.plot(10000. * np.array(history.history["tt_output_loss"]), label = "tt_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["ze_output_loss"]), label = "ze_output_loss", linewidth = 1)
        plt.ylim((0.01, 1000))
        plt.yscale('log')
    plt.title("total {} samples\n effective {} samples\ntrain on {} samples\nvalidate on {} samples\ntest on {} samples".format(totNum, effNum, len(y_train), len(y_val), len(y_test)))
    plt.legend()
    plt.grid(True)
    plt.xlabel("epochs")
    plt.ylabel("{}_loss".format(pred))
    plt.tight_layout()
    plt.savefig("{}/loss_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, pred))
    plt.clf()

def plotImportance(model, x, y):
    # plot importance of each feature based on permutation feature importance
    score_before = model.evaluate(x, [y_test[:, 0], y_test[:, 4], y_test[:, 5], y_test[:, 6], y_test[:, 8], y_test[:, 10], y_test[:, 11], y_test[:, 12]], batch_size = batch, verbose = 0)
    test_pred = np.array(model.predict(x, batch_size = batch))
    pp_test = np.degrees(y_test[:, 3].reshape((len(y_test), 1)))
    pp_test_pred = np.arctan2(test_pred[3], test_pred[2])
    pp_test_pred = np.degrees(np.where(pp_test_pred < 0, pp_test_pred + 2 * np.pi, pp_test_pred))
    score_before.append(np.mean((pp_test - pp_test_pred) ** 2.)) # add mse of vertex azimuth
    az_test = np.degrees(y_test[:, 7].reshape((len(y_test), 1)))
    az_test_pred = np.arctan2(test_pred[6], test_pred[5])
    az_test_pred = np.degrees(np.where(az_test_pred < 0, az_test_pred + 2 * np.pi, az_test_pred))
    score_before.append(np.mean((az_test - az_test_pred) ** 2.)) # add mse of neutrino azimuth
    score_before.append(np.mean(((az_test - pp_test) - (az_test_pred - pp_test_pred)) ** 2.)) # add mse of difference between neutrino and vertex azimuth
    importance = []
    for i in range(x.shape[3]):
        for j in range(x.shape[2]):
            for k in range(x.shape[1]):
                origin = deepcopy(x[:, k, j, i])
                np.random.shuffle(x[:, k, j, i]) # shuffle features
                score_after = model.evaluate(x, [y_test[:, 0], y_test[:, 4], y_test[:, 5], y_test[:, 6], y_test[:, 8], y_test[:, 10], y_test[:, 11], y_test[:, 12]], batch_size = batch, verbose = 0)
                test_pred = np.array(model.predict(x, batch_size = batch))
                pp_test_pred = np.arctan2(test_pred[3], test_pred[2])
                pp_test_pred = np.degrees(np.where(pp_test_pred < 0, pp_test_pred + 2 * np.pi, pp_test_pred))
                score_after.append(np.mean((pp_test - pp_test_pred) ** 2.))
                az_test_pred = np.arctan2(test_pred[6], test_pred[5])
                az_test_pred = np.degrees(np.where(az_test_pred < 0, az_test_pred + 2 * np.pi, az_test_pred))
                score_after.append(np.mean((az_test - az_test_pred) ** 2.))
                score_after.append(np.mean(((az_test - pp_test) - (az_test_pred - pp_test_pred)) ** 2.))
                importance.append([(l - m) / m for l, m in zip(score_after, score_before)]) # calculated the difference of loss by shuffling features
                x[:, k, j, i] = deepcopy(origin)
    importance = np.array(importance)
    sbn.heatmap(importance, norm = LogNorm(vmin = 0.01, vmax = 100), linewidths = .5, linecolor = "black", cmap = cm.Blues)
    xlabel = ["total", "rr", "tt", "cos", "sin", "ze", "cosAz", "sinAz", "sh", "pp", "az", "az - pp"]
    plt.xticks(range(12), xlabel, rotation = "vertical")
    # in case of different input structures
    if x.shape[3] == 3:
        ylabel = ["t", "a", "r"]
    elif x.shape[3] == 4:
        ylabel = ["t_dir", "t_ref", "a_dir", "a_ref"]
    elif x.shape[3] == 6:
        ylabel = ["t_dir", "t_ref", "a_dir", "a_ref", "r_dir", "r_ref"]
    elif x.shape[3] == 8:
        ylabel = ["t_dir", "t_ref", "a_dir", "a_ref", "r_dir", "r_ref", "f_dir", "f_ref"]
    plt.yticks(range(0, x.shape[1] * x.shape[2] * x.shape[3], x.shape[1] * x.shape[2]), ylabel, rotation = "vertical")
    plt.tight_layout()
    plt.savefig("{}/importance_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor))
    plt.clf()

testFile = 15 # the first testFile files are used as test sets
#testFile = 2
shareConv = 1 # all branchs share conv layers 1, not share 0
numSplit = 20 # 20 fold cross validation by default, 95% training, 5% validation
strNum = 4 # number of strings for a detector
channelPerStr = 4 # number of antennas on each string
sampling_rate = 1.6 # in GHz
#random_seed = 1 # global random seed
#Vrms = 9.6297e-06 # in V
Vrms = 3e-6 # in V
lr = 1e-3 # start learning rate
d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0 # dropout rate for input, conv, flatten, dense layers
Energies = []
timeNoiseFactor = float(sys.argv[-1])
ampNoiseFactor = float(sys.argv[-2])
fold = int(sys.argv[-3]) # the foldth fold as validation and test, unless doing cross validation, always use the 0th fold
if fold > numSplit - 1:
    sys.exit("Input error! No \"{}th\" fold in {} fold xvalidation!".format(fold, numSplit))
batch = int(sys.argv[-4]) # batch size
epochs = int(sys.argv[-5]) # number of epochs for training
nodes = int(sys.argv[-6]) # number of nodes for each layer, constant for all the layers
layers = int(sys.argv[-7]) # number of layers for convolutional and fully connected parts, this is number for each part
inMode = int(sys.argv[-8]) # 0: read in NuRadioMC hdf5 files, 1: read in npy files, 2: read in csv files, 3: self defined format
Pred = sys.argv[-9] # flag for functionality
if Pred not in ["train", "rr", "tt", "pp", "ze", "az", "sh"]:
    sys.exit("Input error! No argument \"{}\"!".format(Pred))
outPath = sys.argv[-10]
if not os.path.exists(outPath):
    os.makedirs(outPath)
t1 = time.time()
print("Finished importing package in {:.3f}s".format(t1 - t0))
t0 = time.time()
print("Config: output path: {}\ninfile mode: {}, 2 * {} layers, {} nodes each layer, train for {} epochs, {} samples each batch, {}th fold in {} folds, {} Vrms amp noise, {} bin time noise".format(outPath, inMode, layers, nodes, epochs, batch, fold, numSplit, ampNoiseFactor, timeNoiseFactor))
print("Modeling: {}".format(Pred))
if inMode == 0:
    # read in hdf5 from NuRadioMC1.x
    inFile = h5py.File(sys.argv[1], 'r') # start to read in files
    print("Reading " + str(sys.argv[1]))
    event_ids = np.array(inFile['event_ids'])
    xx = np.array(inFile['xx'])
    yy = np.array(inFile['yy'])
    zz = np.array(inFile['zz'])
    azimuths = np.array(inFile['azimuths'])
    zeniths = np.array(inFile['zeniths'])
    energies = np.array(inFile['energies'])
    if np.round(np.log10(inFile['energies'][0]), 1) not in Energies:
        Energies.append(np.round(np.log10(inFile['energies'][0]), 1))
    flavors = np.array(inFile['flavors'])
    inelasticity = np.array(inFile['inelasticity'])
    interaction_type = np.array(inFile['interaction_type'])
    SNRs = np.array(inFile['station_101']['SNRs'])
    max_amp_ray_solution = np.array(inFile['station_101']['max_amp_ray_solution'])
    max_amp_envelope = np.array(inFile['station_101']['max_amp_envelope'])
    max_amp_phi = np.array(inFile['station_101']['max_amp_phi'])
    max_amp_phi_envelope = np.array(inFile['station_101']['max_amp_phi_envelope'])
    max_amp_theta = np.array(inFile['station_101']['max_amp_theta'])
    max_amp_theta_envelope = np.array(inFile['station_101']['max_amp_theta_envelope'])
    max_freq_ray_solution = np.array(inFile['station_101']['max_freq_ray_solution'])
    travel_times = np.array(inFile['station_101']['travel_times'])
    receive_vectors = np.array(inFile['station_101']['receive_vectors'])
    launch_vectors = np.array(inFile['station_101']['launch_vectors'])
    polarization = np.array(inFile['station_101']['polarization'])
    testLen = event_ids.shape[0] # use the first testFile files as test set
    testFile -= 1
    print("as test file")
    for i in range(2, len(sys.argv) - 10):
        inFile = h5py.File(sys.argv[i], 'r')
        print("Reading " + str(sys.argv[i]))
        event_ids = np.append(event_ids, np.array(inFile['event_ids']))
        xx = np.append(xx, np.array(inFile['xx']))
        yy = np.append(yy, np.array(inFile['yy']))
        zz = np.append(zz, np.array(inFile['zz']))
        azimuths = np.append(azimuths, np.array(inFile['azimuths']))
        zeniths = np.append(zeniths, np.array(inFile['zeniths']))
        energies = np.append(energies, np.array(inFile['energies']))
        if np.round(np.log10(inFile['energies'][0]), 1) not in Energies:
            Energies.append(np.round(np.log10(inFile['energies'][0]), 1))
        flavors = np.append(flavors, np.array(inFile['flavors']))
        inelasticity = np.append(inelasticity, np.array(inFile['inelasticity']))
        interaction_type = np.append(interaction_type, np.array(inFile['interaction_type']))
        SNRs = np.append(SNRs, np.array(inFile['station_101']['SNRs']))
        max_amp_ray_solution = np.append(max_amp_ray_solution, np.array(inFile['station_101']['max_amp_ray_solution']), axis = 0)
        max_amp_envelope = np.append(max_amp_envelope, np.array(inFile['station_101']['max_amp_envelope']), axis = 0)
        max_amp_phi = np.append(max_amp_phi, np.array(inFile['station_101']['max_amp_phi']), axis = 0)
        max_amp_phi_envelope = np.append(max_amp_phi_envelope, np.array(inFile['station_101']['max_amp_phi_envelope']), axis = 0)
        max_amp_theta = np.append(max_amp_theta, np.array(inFile['station_101']['max_amp_theta']), axis = 0)
        max_amp_theta_envelope = np.append(max_amp_theta_envelope, np.array(inFile['station_101']['max_amp_theta_envelope']), axis = 0)
        max_freq_ray_solution = np.append(max_freq_ray_solution, np.array(inFile['station_101']['max_freq_ray_solution']), axis = 0)
        travel_times = np.append(travel_times, np.array(inFile['station_101']['travel_times']), axis = 0)
        receive_vectors = np.append(receive_vectors, np.array(inFile['station_101']['receive_vectors']), axis = 0)
        launch_vectors = np.append(launch_vectors, np.array(inFile['station_101']['launch_vectors']), axis = 0)
        polarization = np.append(polarization, np.array(inFile['station_101']['polarization']), axis = 0)
        if testFile > 0: 
            testLen = event_ids.shape[0] # use the first testFile file as test set
            testFile -= 1
            print("as test file")
        else:
            print("as training file")
elif inMode == 1:
    # read in file from npy in the format of:
    # evtId, xx, yy, zz, az, ze, energies, flavors, inelasticities, interaction type, snr, peak volts for all channels and rays, peak envelope for all channels and rays, peak hpol ef for all channels and rays, peak envelope hpol ef for all channels and rays, peak vpol ef for all channels and rays, peak envelope vpol ef for all channels and rays, peak freq for all channels and rays, travel times for all channels and rays, rec vectors in cartesian coordinate sys for all channels and rays, launch vectors in cartesian coordinate sys for all channels and rays, pol vectors in cartesian coordinate sys for all channels and rays
    inFile = np.load(sys.argv[1]) # start to read in files
    print("Reading " + str(sys.argv[1]))
    evtNum = len(inFile[:, 0])
    event_ids = inFile[:, 0]
    xx = inFile[:, 1]
    yy = inFile[:, 2]
    zz = inFile[:, 3]
    azimuths = inFile[:, 4]
    zeniths = inFile[:, 5]
    energies = inFile[:, 6]
    if np.round(np.log10(inFile[0, 6]), 1) not in Energies:
        Energies.append(np.round(np.log10(inFile[0, 6]), 1))
    flavors = inFile[:, 7]
    inelasticity = inFile[:, 8]
    interaction_type = inFile[:, 9]
    SNRs = inFile[:, 10]
    max_amp_ray_solution = inFile[:, 11:(11 + channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    max_amp_envelope = inFile[:, (11 + channelPerStr * strNum * 2):(11 + 2 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    max_amp_phi = inFile[:, (11 + 2 * channelPerStr * strNum * 2):(11 + 3 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    max_amp_phi_envelope = inFile[:, (11 + 3 * channelPerStr * strNum * 2):(11 + 4 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    max_amp_theta = inFile[:, (11 + 4 * channelPerStr * strNum * 2):(11 + 5 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    max_amp_theta_envelope = inFile[:, (11 + 5 * channelPerStr * strNum * 2):(11 + 6 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    max_freq_ray_solution = inFile[:, (11 + 6 * channelPerStr * strNum * 2):(11 + 7 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    travel_times = inFile[:, (11 + 7 * channelPerStr * strNum * 2):(11 + 8 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2)
    receive_vectors = inFile[:, (11 + 8 * channelPerStr * strNum * 2):(11 + 8 * channelPerStr * strNum * 2 + channelPerStr * strNum * 2 * 3)].reshape(evtNum, channelPerStr * strNum, 2, 3)
    launch_vectors = inFile[:, (11 + 8 * channelPerStr * strNum * 2 + channelPerStr * strNum * 2 * 3):(11 + 8 * channelPerStr * strNum * 2 + 2 * channelPerStr * strNum * 2 * 3)].reshape(evtNum, channelPerStr * strNum, 2, 3)
    polarization = inFile[:, (11 + 8 * channelPerStr * strNum * 2 + 2 * channelPerStr * strNum * 2 * 3):(11 + 8 * channelPerStr * strNum * 2 + 3 * channelPerStr * strNum * 2 * 3)].reshape(evtNum, channelPerStr * strNum, 2, 3)
    unixTime = inFile[:, -1]
    testLen = event_ids.shape[0] # use the first file as test set
    testFile -= 1
    print("as test file")
    for i in range(2, len(sys.argv) - 10):
        inFile = np.load(sys.argv[i])
        print("Reading " + str(sys.argv[i]))
        evtNum = len(inFile[:, 0])
        event_ids = np.append(event_ids, inFile[:, 0])
        xx = np.append(xx, inFile[:, 1])
        yy = np.append(yy, inFile[:, 2])
        zz = np.append(zz, inFile[:, 3])
        azimuths = np.append(azimuths, inFile[:, 4])
        zeniths = np.append(zeniths, inFile[:, 5])
        energies = np.append(energies, inFile[:, 6])
        if np.round(np.log10(inFile[0, 6]), 1) not in Energies:
            Energies.append(np.round(np.log10(inFile[0, 6]), 1))
        flavors = np.append(flavors, inFile[:, 7])
        inelasticity = np.append(inelasticity, inFile[:, 8])
        interaction_type = np.append(interaction_type, inFile[:, 9])
        SNRs = np.append(SNRs, inFile[:, 10])
        max_amp_ray_solution = np.append(max_amp_ray_solution, inFile[:, 11:(11 + channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        max_amp_envelope = np.append(max_amp_envelope, inFile[:, (11 + channelPerStr * strNum * 2):(11 + 2 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        max_amp_phi = np.append(max_amp_phi, inFile[:, (11 + 2 * channelPerStr * strNum * 2):(11 + 3 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        max_amp_phi_envelope = np.append(max_amp_phi_envelope, inFile[:, (11 + 3 * channelPerStr * strNum * 2):(11 + 4 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        max_amp_theta = np.append(max_amp_theta, inFile[:, (11 + 4 * channelPerStr * strNum * 2):(11 + 5 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        max_amp_theta_envelope = np.append(max_amp_theta_envelope, inFile[:, (11 + 5 * channelPerStr * strNum * 2):(11 + 6 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        max_freq_ray_solution = np.append(max_freq_ray_solution, inFile[:, (11 + 6 * channelPerStr * strNum * 2):(11 + 7 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        travel_times = np.append(travel_times, inFile[:, (11 + 7 * channelPerStr * strNum * 2):(11 + 8 * channelPerStr * strNum * 2)].reshape(evtNum, channelPerStr * strNum, 2), axis = 0)
        receive_vectors = np.append(receive_vectors, inFile[:, (11 + 8 * channelPerStr * strNum * 2):(11 + 8 * channelPerStr * strNum * 2 + channelPerStr * strNum * 2 * 3)].reshape(evtNum, channelPerStr * strNum, 2, 3), axis = 0)
        launch_vectors = np.append(launch_vectors, inFile[:, (11 + 8 * channelPerStr * strNum * 2 + channelPerStr * strNum * 2 * 3):(11 + 8 * channelPerStr * strNum * 2 + 2 * channelPerStr * strNum * 2 * 3)].reshape(evtNum, channelPerStr * strNum, 2, 3), axis = 0)
        polarization = np.append(polarization, inFile[:, (11 + 8 * channelPerStr * strNum * 2 + 2 * channelPerStr * strNum * 2 * 3):(11 + 8 * channelPerStr * strNum * 2 + 3 * channelPerStr * strNum * 2 * 3)].reshape(evtNum, channelPerStr * strNum, 2, 3), axis = 0)
        unixTime = np.append(unixTime, inFile[:, -1], axis = 0)
        if testFile > 0:
            testLen = event_ids.shape[0] # use the first testFile file as test set
            testFile -= 1
            print("as test file")
        else:
            print("as training file")
elif inMode == 2:
    # read in file from csv in the format of:
    # evtId, xx, yy, zz, az, ze, energies, flavors, inelasticities, interaction type, snr, peak volts for all channels and rays, peak envelope for all channels and rays, peak hpol ef for all channels and rays, peak envelope hpol ef for all channels and rays, peak vpol ef for all channels and rays, peak envelope vpol ef for all channels and rays, peak freq for all channels and rays, travel times for all channels and rays, rec vectors in cartesian coordinate sys for all channels and rays, launch vectors in cartesian coordinate sys for all channels and rays, pol vectors in cartesian coordinate sys for all channels and rays
    sys.exit("Read in mode 2 has not been supported yet!")
elif inMode == 3:
    # self defined read in format
    sys.exit("Read in mode 3 has not been supported yet!")
else:
    sys.exit("Input error! Read in mode {} not valid!".format(inMode))

t1 = time.time()
print("Finished reading files in {:.3f}s".format(t1 - t0))
t0 = time.time()
print("Data preparing ...")
Energies.sort()
evtNum = len(event_ids)
interaction = np.zeros((evtNum, ), dtype = int)
# turn interaction type into 0/1 for nc/cc
'''
for i in range(evtNum):
    if interaction_type[i] == b'cc':
        interaction[i] = 1
'''
interaction = np.array([1 if i == b'cc' else 0 for i in interaction_type])
#calculate viewing angle
shower_axis = -1.0 * hp.spherical_to_cartesian(zeniths, azimuths)
view_angles = np.zeros((evtNum, channelPerStr * strNum, 2))
view_angles[:, 0, 0] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])
view_angles[:, 0, 1] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 1])])
for i in range(1, channelPerStr * strNum):
    view_angles[:, i, 0] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, i, 0])])
    view_angles[:, i, 1] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, i, 1])])
# calculate cone angle
cone_angles = np.zeros((evtNum, channelPerStr * strNum, 2))
xHat = np.array([1., 0., 0.])
yHat = np.array([0., 1., 0.])
zHat = np.array([0., 0., 1.])
launch_vectors_dir = launch_vectors[:, 0, 0, :]
launch_vectors_ref = launch_vectors[:, 0, 1, :]
yOnCone = np.cross(zHat, shower_axis)
yOnCone = yOnCone / np.linalg.norm(yOnCone, axis = 1).reshape((yOnCone.shape[0], 1))
zOnCone = np.cross(shower_axis, yOnCone)
zOnCone = zOnCone / np.linalg.norm(zOnCone, axis = 1).reshape((zOnCone.shape[0], 1))
launchOnCone_dir = launch_vectors_dir - shower_axis * np.sum(launch_vectors_dir * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
launchOnCone_dir = launchOnCone_dir / np.linalg.norm(launchOnCone_dir, axis = 1).reshape((launchOnCone_dir.shape[0], 1))
cone_angles[:, 0, 0] = np.arccos(np.sum(launchOnCone_dir * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_dir * yOnCone, axis = 1))
launchOnCone_ref = launch_vectors_ref - shower_axis * np.sum(launch_vectors_ref * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
launchOnCone_ref = launchOnCone_ref / np.linalg.norm(launchOnCone_ref, axis = 1).reshape((launchOnCone_ref.shape[0], 1))
cone_angles[:, 0, 1] = np.arccos(np.sum(launchOnCone_ref * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_ref * yOnCone, axis = 1))
for i in range(1, channelPerStr * strNum):
    launch_vectors_dir = launch_vectors[:, i, 0, :]
    launch_vectors_ref = launch_vectors[:, i, 1, :]
    yOnCone = np.cross(zHat, shower_axis)
    yOnCone = yOnCone / np.linalg.norm(yOnCone, axis = 1).reshape((yOnCone.shape[0], 1))
    zOnCone = np.cross(shower_axis, yOnCone)
    zOnCone = zOnCone / np.linalg.norm(zOnCone, axis = 1).reshape((zOnCone.shape[0], 1))
    launchOnCone_dir = launch_vectors_dir - shower_axis * np.sum(launch_vectors_dir * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
    launchOnCone_dir = launchOnCone_dir / np.linalg.norm(launchOnCone_dir, axis = 1).reshape((launchOnCone_dir.shape[0], 1))
    cone_angles[:, i, 0] = np.arccos(np.sum(launchOnCone_dir * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_dir * yOnCone, axis = 1))
    launchOnCone_ref = launch_vectors_ref - shower_axis * np.sum(launch_vectors_ref * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
    launchOnCone_ref = launchOnCone_ref / np.linalg.norm(launchOnCone_ref, axis = 1).reshape((launchOnCone_ref.shape[0], 1))
    cone_angles[:, i, 1] = np.arccos(np.sum(launchOnCone_ref * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_ref * yOnCone, axis = 1))
# calculate cherenkov angle
ice = medium.southpole_2015()
n_index = np.array([ice.get_index_of_refraction(x) for x in np.array([xx, yy, zz]).T])
cherenkov = np.arccos(1. / n_index)
#calculate polarization angle
yOnCone = np.cross(zHat, receive_vectors)
yOnCone = yOnCone / np.linalg.norm(yOnCone, axis = 3).reshape(yOnCone.shape[0], yOnCone.shape[1], yOnCone.shape[2], 1)
zOnCone = np.cross(receive_vectors, yOnCone)
zOnCone = zOnCone / np.linalg.norm(zOnCone, axis = 3).reshape(zOnCone.shape[0], zOnCone.shape[1], zOnCone.shape[2], 1)
hAmp = np.sum(polarization * yOnCone, axis = -1)
vAmp = np.sum(polarization * zOnCone, axis = -1)
pol_angles = np.arctan2(hAmp, vAmp)
pol_angles = np.where(pol_angles < 0, pol_angles + 2. * np.pi, pol_angles) # turn into 0-2pi
rec_theta = np.arccos(np.sum(receive_vectors * zHat, axis = -1))
rec_phi = np.arctan2(np.sum(receive_vectors * yHat, axis = -1), np.sum(receive_vectors * xHat, axis = -1))
rec_phi = np.where(rec_phi < 0, rec_phi + 2. * np.pi, rec_phi)
pol_theta = np.arccos(np.sum(polarization * zHat, axis = -1))
pol_phi = np.arctan2(np.sum(polarization * yHat, axis = -1), np.sum(polarization * xHat, axis = -1))
pol_phi = np.where(pol_phi < 0, pol_phi + 2. * np.pi, pol_phi)
showerEnergies = np.round(np.log10(np.where((np.abs(flavors) == 12) & (interaction == 1), 1., inelasticity) * energies), 2) # calculate shower energy
energies = np.round(np.log10(energies), 1)
zz = np.absolute(zz) #use positive depth as under surface
rr = np.sqrt(np.square(xx) + np.square(yy)) # horizontal distance from detector center to vertex
tt = -1. * np.arcsin((zz - 200.) / np.sqrt(np.square(xx) + np.square(yy) + np.square(zz - 200.))) # vertex zenith angle from detector center and 200m deep
pp = np.arctan2(yy, xx)
pp = np.where(pp < 0, pp + 2 * np.pi, pp) # vertex azimuth 0-2pi
dd = np.sqrt(np.square(rr) + np.square(zz - 200.)) # vertex distance from detector center and 200m deep
rt, rp = hp.cartesian_to_spherical(receive_vectors[:, :, 0, 0].flatten(), receive_vectors[:, :, 0, 1].flatten(), receive_vectors[:, :, 0, 2].flatten())#receive_vectors[evt, chan, dir/ref, xyz]
rt = -1. * np.mean(rt.reshape(receive_vectors.shape[0], channelPerStr * strNum), axis = 1) + np.pi / 2.
cos = xx / rr # vertex cos azimuth
sin = yy / rr # vertex sin azimuth
cosAz = np.cos(azimuths) # neutrino cos azimuth
sinAz = np.sin(azimuths) # neutrino sin azimuth

t1 = time.time()
print("Finished data preparation in {:.3f}s".format(t1 - t0))
t0 = time.time()
print("Data cleaning ...")

travel_times = normalTimeDiff(travel_times)
travel_times = travel_times.reshape(travel_times.shape[0] * channelPerStr * strNum, 2)
Filter = travel_times[:, 0] < travel_times[:, 1] # always put the ray with shorter travel time first
travel_times_dir = np.where(Filter, travel_times[:, 0], travel_times[:, 1])
travel_times_ref = np.where(Filter, travel_times[:, 1], travel_times[:, 0])
# turn into a 4*4 array with columns representing strings and rows representing antennas in topV botV topH botH
travel_times_dir = travel_times_dir.reshape(int(travel_times_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
travel_times_ref = travel_times_ref.reshape(int(travel_times_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
if timeNoiseFactor != 0.:  # apply normally distributed noise
    print("Adding time noise N(0.0, {}) bin size ...".format(timeNoiseFactor))
    travel_times_dir += np.random.normal(0., timeNoiseFactor * 1. / sampling_rate, travel_times_dir.shape)
    travel_times_ref += np.random.normal(0., timeNoiseFactor * 1. / sampling_rate, travel_times_ref.shape)
max_amp_ray_solution = np.absolute(max_amp_ray_solution) # use absolute value of peak
max_amp_ray_solution = max_amp_ray_solution.reshape(max_amp_ray_solution.shape[0] * channelPerStr * strNum, 2)
max_amp_ray_solution_dir = np.where(Filter, max_amp_ray_solution[:, 0], max_amp_ray_solution[:, 1])
max_amp_ray_solution_ref = np.where(Filter, max_amp_ray_solution[:, 1], max_amp_ray_solution[:, 0])
max_amp_ray_solution_dir = max_amp_ray_solution_dir.reshape(int(max_amp_ray_solution_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_ray_solution_ref = max_amp_ray_solution_ref.reshape(int(max_amp_ray_solution_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_phi = np.absolute(max_amp_phi) # use absolute value of horizontally polarized ef
max_amp_phi = max_amp_phi.reshape(max_amp_phi.shape[0] * channelPerStr * strNum, 2)
max_amp_phi_dir = np.where(Filter, max_amp_phi[:, 0], max_amp_phi[:, 1])
max_amp_phi_ref = np.where(Filter, max_amp_phi[:, 1], max_amp_phi[:, 0])
max_amp_phi_dir = max_amp_phi_dir.reshape(int(max_amp_phi_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_phi_ref = max_amp_phi_ref.reshape(int(max_amp_phi_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_phi_envelope = np.absolute(max_amp_phi_envelope) # use absolute value of horizontally polarized ef envelope
max_amp_phi_envelope = max_amp_phi_envelope.reshape(max_amp_phi_envelope.shape[0] * channelPerStr * strNum, 2)
max_amp_phi_envelope_dir = np.where(Filter, max_amp_phi_envelope[:, 0], max_amp_phi_envelope[:, 1])
max_amp_phi_envelope_ref = np.where(Filter, max_amp_phi_envelope[:, 1], max_amp_phi_envelope[:, 0])
max_amp_phi_envelope_dir = max_amp_phi_envelope_dir.reshape(int(max_amp_phi_envelope_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_phi_envelope_ref = max_amp_phi_envelope_ref.reshape(int(max_amp_phi_envelope_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_theta = np.absolute(max_amp_theta) # use absolute value of vertically polarized ef
max_amp_theta = max_amp_theta.reshape(max_amp_theta.shape[0] * channelPerStr * strNum, 2)
max_amp_theta_dir = np.where(Filter, max_amp_theta[:, 0], max_amp_theta[:, 1])
max_amp_theta_ref = np.where(Filter, max_amp_theta[:, 1], max_amp_theta[:, 0])
max_amp_theta_dir = max_amp_theta_dir.reshape(int(max_amp_theta_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_theta_ref = max_amp_theta_ref.reshape(int(max_amp_theta_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_theta_envelope = np.absolute(max_amp_theta_envelope) # use absolute value of vertically polarized ef envelope
max_amp_theta_envelope = max_amp_theta_envelope.reshape(max_amp_theta_envelope.shape[0] * channelPerStr * strNum, 2)
max_amp_theta_envelope_dir = np.where(Filter, max_amp_theta_envelope[:, 0], max_amp_theta_envelope[:, 1])
max_amp_theta_envelope_ref = np.where(Filter, max_amp_theta_envelope[:, 1], max_amp_theta_envelope[:, 0])
max_amp_theta_envelope_dir = max_amp_theta_envelope_dir.reshape(int(max_amp_theta_envelope_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_theta_envelope_ref = max_amp_theta_envelope_ref.reshape(int(max_amp_theta_envelope_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_freq_ray_solution = max_freq_ray_solution.reshape(max_freq_ray_solution.shape[0] * channelPerStr * strNum, 2)
max_freq_ray_solution_dir = np.where(Filter, max_freq_ray_solution[:, 0], max_freq_ray_solution[:, 1])
max_freq_ray_solution_ref = np.where(Filter, max_freq_ray_solution[:, 1], max_freq_ray_solution[:, 0])
max_freq_ray_solution_dir = max_freq_ray_solution_dir.reshape(int(max_freq_ray_solution_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_freq_ray_solution_ref = max_freq_ray_solution_ref.reshape(int(max_freq_ray_solution_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_freq_ray_solution_dir = np.log10(max_freq_ray_solution_dir)
max_freq_ray_solution_ref = np.log10(max_freq_ray_solution_ref)
pol_angles = pol_angles.reshape(pol_angles.shape[0] * channelPerStr * strNum, 2)
pol_angles_dir = np.where(Filter, pol_angles[:, 0], pol_angles[:, 1])
pol_angles_ref = np.where(Filter, pol_angles[:, 1], pol_angles[:, 0])
pol_angles_dir = pol_angles_dir.reshape(int(pol_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
pol_angles_ref = pol_angles_ref.reshape(int(pol_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
hAmp = hAmp.reshape(hAmp.shape[0] * channelPerStr * strNum, 2)
hAmp_dir = np.where(Filter, hAmp[:, 0], hAmp[:, 1])
hAmp_ref = np.where(Filter, hAmp[:, 1], hAmp[:, 0])
hAmp_dir = hAmp_dir.reshape(int(hAmp_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
hAmp_ref = hAmp_ref.reshape(int(hAmp_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
vAmp = vAmp.reshape(vAmp.shape[0] * channelPerStr * strNum, 2)
vAmp_dir = np.where(Filter, vAmp[:, 0], vAmp[:, 1])
vAmp_ref = np.where(Filter, vAmp[:, 1], vAmp[:, 0])
vAmp_dir = vAmp_dir.reshape(int(vAmp_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
vAmp_ref = vAmp_ref.reshape(int(vAmp_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
'''
# apply true sign to amp to calculate ratio
max_amp_ray_solution_dir[:, 0:2, :] *= np.sign(vAmp_dir[:, 0:2, :])
max_amp_ray_solution_dir[:, 2:4, :] *= np.sign(hAmp_dir[:, 2:4, :])
max_amp_ray_solution_ref[:, 0:2, :] *= np.sign(vAmp_ref[:, 0:2, :])
max_amp_ray_solution_ref[:, 2:4, :] *= np.sign(hAmp_ref[:, 2:4, :])
'''
# apply normally distributed noise
if ampNoiseFactor != 0.:
    print("Adding amp noise N(0.0, {}) Vrms ...".format(ampNoiseFactor))
    max_amp_ray_solution_dir += np.random.normal(0., ampNoiseFactor * Vrms, max_amp_ray_solution_dir.shape)
    max_amp_ray_solution_ref += np.random.normal(0., ampNoiseFactor * Vrms, max_amp_ray_solution_ref.shape)
# calculate amp ratio angle 0-2pi
ratio_amp_dir = np.arctan2(max_amp_ray_solution_dir[:, 2:4, :], max_amp_ray_solution_dir[:, 0:2, :])
ratio_amp_dir = np.where(ratio_amp_dir < 0, ratio_amp_dir + 2. * np.pi, ratio_amp_dir)
ratio_amp_ref = np.arctan2(max_amp_ray_solution_ref[:, 2:4, :], max_amp_ray_solution_ref[:, 0:2, :])
ratio_amp_ref = np.where(ratio_amp_ref < 0, ratio_amp_ref + 2. * np.pi, ratio_amp_ref)
# use the same angles for hpol and vpol in 4*4 array
ratio_amp_dir = np.repeat(ratio_amp_dir, 2, axis = 0).reshape(ratio_amp_dir.shape[0], channelPerStr, strNum)
ratio_amp_ref = np.repeat(ratio_amp_ref, 2, axis = 0).reshape(ratio_amp_ref.shape[0], channelPerStr, strNum)
# turn amps back to absolute values
max_amp_ray_solution_dir = np.absolute(max_amp_ray_solution_dir)
max_amp_ray_solution_ref = np.absolute(max_amp_ray_solution_ref)
# recalculate snr in the case of noise
if ampNoiseFactor != 0.:
    SNRs = np.maximum(np.max(max_amp_ray_solution_dir / (ampNoiseFactor * Vrms), axis = (1, 2)), np.max(max_amp_ray_solution_ref / (ampNoiseFactor * Vrms), axis = (1, 2)))
max_amp_ray_solution_dir = np.log10(max_amp_ray_solution_dir)
max_amp_ray_solution_ref = np.log10(max_amp_ray_solution_ref)
# apply true sign to ef
max_amp_theta_dir[:, 0:2, :] = max_amp_theta_dir[:, 0:2, :] * np.sign(vAmp_dir[:, 0:2, :])
max_amp_phi_dir[:, 2:4, :] = max_amp_phi_dir[:, 2:4, :] * np.sign(hAmp_dir[:, 2:4, :])
max_amp_theta_ref[:, 0:2, :] = max_amp_theta_ref[:, 0:2, :] * np.sign(vAmp_ref[:, 0:2, :])
max_amp_phi_ref[:, 2:4, :] = max_amp_phi_ref[:, 2:4, :] * np.sign(hAmp_ref[:, 2:4, :])
# calculate ef ratio angle 0-2pi
ratio_ef_dir = np.arctan2(max_amp_phi_dir[:, 2:4, :], max_amp_theta_dir[:, 0:2, :])
ratio_ef_dir = np.where(ratio_ef_dir < 0, ratio_ef_dir + 2. * np.pi, ratio_ef_dir)
ratio_ef_ref = np.arctan2(max_amp_phi_ref[:, 2:4, :], max_amp_theta_ref[:, 0:2, :])
ratio_ef_ref = np.where(ratio_ef_ref < 0, ratio_ef_ref + 2. * np.pi, ratio_ef_ref)
ratio_ef_dir = np.repeat(ratio_ef_dir, 2, axis = 0).reshape(ratio_ef_dir.shape[0], channelPerStr, strNum)
ratio_ef_ref = np.repeat(ratio_ef_ref, 2, axis = 0).reshape(ratio_ef_ref.shape[0], channelPerStr, strNum)
# reshape them to apply Filter
cone_angles = cone_angles.reshape(cone_angles.shape[0] * channelPerStr * strNum, 2)
view_angles = view_angles.reshape(view_angles.shape[0] * channelPerStr * strNum, 2)
# put rays with shorter travel times first
cone_angles_dir = np.where(Filter, cone_angles[:, 0], cone_angles[:, 1])
cone_angles_ref = np.where(Filter, cone_angles[:, 1], cone_angles[:, 0])
view_angles_dir = np.where(Filter, view_angles[:, 0], view_angles[:, 1])
view_angles_ref = np.where(Filter, view_angles[:, 1], view_angles[:, 0])
# turn them back to event*antenna*str
cone_angles_dir = cone_angles_dir.reshape(int(cone_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
cone_angles_ref = cone_angles_ref.reshape(int(cone_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
view_angles_dir = view_angles_dir.reshape(int(view_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
view_angles_ref = view_angles_ref.reshape(int(view_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
# use first channel as metric
viewAngle_dir = deepcopy(view_angles_dir[:, 0, 0])
viewAngle_ref = deepcopy(view_angles_ref[:, 0, 0])
coneAngle_dir = deepcopy(cone_angles_dir[:, 0, 0])
coneAngle_ref = deepcopy(cone_angles_ref[:, 0, 0])
# create a general x array
x = np.stack((travel_times_dir, travel_times_ref, max_amp_ray_solution_dir, max_amp_ray_solution_ref, ratio_amp_dir, ratio_amp_ref, max_freq_ray_solution_dir, max_freq_ray_solution_ref, view_angles_dir, view_angles_ref, cone_angles_dir, cone_angles_ref), axis = 3)

# create a general y array
if inMode == 0:
    y = np.vstack((rr, zz, dd, pp, tt, cos, sin, azimuths, zeniths, energies, cosAz, sinAz, showerEnergies, xx, yy, flavors, viewAngle_dir, viewAngle_ref, coneAngle_dir, coneAngle_ref, SNRs, event_ids))
elif inMode == 1:
    y = np.vstack((rr, zz, dd, pp, tt, cos, sin, azimuths, zeniths, energies, cosAz, sinAz, showerEnergies, xx, yy, flavors, viewAngle_dir, viewAngle_ref, coneAngle_dir, coneAngle_ref, SNRs, event_ids, unixTime))
y = np.transpose(y)
totNum = x.shape[0]
maskY = np.isnan(y).any(axis = 1) # nan filter
maskX = np.isnan(x).any(axis = 1).any(axis = 1).any(axis = 1) # nan filter
maskSNR = np.where(SNRs < 4., True, False) # snr filter
maskEM = np.logical_and(np.logical_or(flavors == 12, flavors == -12), interaction == 1) # em shower filter
maskVertex = np.where((xx > -200.) | (xx < -700.) | (yy > -2200.) | (yy < -2800.), True, False)
# percentile filter
cone_angles_dir = cone_angles_dir.reshape(cone_angles_dir.shape[0] * channelPerStr * strNum, 1)
cone_angles_ref = cone_angles_ref.reshape(cone_angles_ref.shape[0] * channelPerStr * strNum, 1)
maskCone = np.where((cone_angles_dir > np.nanpercentile(cone_angles_dir, 90)) | (cone_angles_dir < np.nanpercentile(cone_angles_dir, 10)) | (cone_angles_ref > np.nanpercentile(cone_angles_ref, 90)) | (cone_angles_ref < np.nanpercentile(cone_angles_ref, 10)), True, False).reshape(int(cone_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum).any(axis = 1).any(axis = 1)
cone_angles_dir = cone_angles_dir.reshape(int(cone_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
cone_angles_ref = cone_angles_ref.reshape(int(cone_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
view_angles_dir = view_angles_dir.reshape(view_angles_dir.shape[0] * channelPerStr * strNum, 1)
view_angles_ref = view_angles_ref.reshape(view_angles_ref.shape[0] * channelPerStr * strNum, 1)
maskView = np.where((view_angles_dir > np.nanpercentile(view_angles_dir, 90)) | (view_angles_dir < np.nanpercentile(view_angles_dir, 10)) | (view_angles_ref > np.nanpercentile(view_angles_ref, 90)) | (view_angles_ref < np.nanpercentile(view_angles_ref, 10)), True, False).reshape(int(view_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum).any(axis = 1).any(axis = 1)
view_angles_dir = view_angles_dir.reshape(int(view_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
view_angles_ref = view_angles_ref.reshape(int(view_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
# finalize a filter
#mask = np.logical_or(np.logical_or(np.logical_or(maskX, maskY), maskCone), maskView)
#mask = np.logical_or(np.logical_or(maskX, maskY), maskSNR)
#mask = np.logical_or(np.logical_or(maskX, maskY), maskEM)
#mask = np.logical_or(maskX, maskY)
#mask = np.logical_or(np.logical_or(maskX, maskY), maskVertex)
mask = maskX
testLen -= np.sum(mask[0:testLen])
x = x[~mask]
y = y[~mask]
effNum = x.shape[0]
efficiency = float(effNum) / float(totNum)
print("{} events in total, {} events left after cleaning, {} events for test,  efficiency = {:.3f}".format(totNum, effNum, testLen, float(effNum) / float(totNum)))

t1 = time.time()
print("Finished data cleaning in {:.3f}s".format(t1 - t0))
t0 = time.time()
print("Spliting data ...")
# use the first few files as test set
x_test = deepcopy(x[0:testLen])
y_test = deepcopy(y[0:testLen])
x = deepcopy(x[testLen:])
y = deepcopy(y[testLen:])
kfold = KFold(n_splits = numSplit, shuffle = True, random_state = random_seed) #split data into numSplit parts
kth = 0
for train, val in kfold.split(x, y):
    if kth != fold: # if not the intended fold num, go to next iteration
        kth += 1
        continue
    x_train = x[train]
    x_val = x[val]
    y_train = y[train]
    y_val = y[val]
    
    if Pred == "train":
        plotFeature(x_test, y_test)
    
    # select from the general x what features to use
    '''
    # timing only
    x_train = np.stack((x_train[:, :, :, 0], x_train[:, :, :, 1]), axis = -1)
    x_val = np.stack((x_val[:, :, :, 0], x_val[:, :, :, 1]), axis = -1)
    x_test = np.stack((x_test[:, :, :, 0], x_test[:, :, :, 1]), axis = -1)
    '''

    # vpol timing only
    x_train = np.stack((x_train[:, 0:2, :, 0], x_train[:, 0:2, :, 1]), axis = -1)
    x_val = np.stack((x_val[:, 0:2, :, 0], x_val[:, 0:2, :, 1]), axis = -1)
    x_test = np.stack((x_test[:, 0:2, :, 0], x_test[:, 0:2, :, 1]), axis = -1)
    
    '''
    # include freq
    x_train = np.stack((x_train[:, :, :, 0], x_train[:, :, :, 1], x_train[:, :, :, 2], x_train[:, :, :, 3], x_train[:, :, :, 12], x_train[:, :, :, 13], x_train[:, :, :, 14], x_train[:, :, :, 15]), axis = -1)
    x_val = np.stack((x_val[:, :, :, 0], x_val[:, :, :, 1], x_val[:, :, :, 2], x_val[:, :, :, 3], x_val[:, :, :, 12], x_val[:, :, :, 13], x_val[:, :, :, 14], x_val[:, :, :, 15]), axis = -1)
    x_test = np.stack((x_test[:, :, :, 0], x_test[:, :, :, 1], x_test[:, :, :, 2], x_test[:, :, :, 3], x_test[:, :, :, 12], x_test[:, :, :, 13], x_test[:, :, :, 14], x_test[:, :, :, 15]), axis = -1)
    '''
    '''
    # no freq
    x_train = np.stack((x_train[:, :, :, 0], x_train[:, :, :, 1], x_train[:, :, :, 2], x_train[:, :, :, 3], x_train[:, :, :, 4], x_train[:, :, :, 5]), axis = -1)
    x_val = np.stack((x_val[:, :, :, 0], x_val[:, :, :, 1], x_val[:, :, :, 2], x_val[:, :, :, 3], x_val[:, :, :, 4], x_val[:, :, :, 5]), axis = -1)
    x_test = np.stack((x_test[:, :, :, 0], x_test[:, :, :, 1], x_test[:, :, :, 2], x_test[:, :, :, 3], x_test[:, :, :, 4], x_test[:, :, :, 5]), axis = -1)
    #x_train[:, :, :, 2] = x_train[:, :, :, 2] / (Vrms ** 2)
    #x_train[:, :, :, 3] = x_train[:, :, :, 3] / (Vrms ** 2)
    '''
    '''
    # no ratio
    x_train = np.stack((x_train[:, :, :, 0], x_train[:, :, :, 1], x_train[:, :, :, 2], x_train[:, :, :, 3]), axis = -1)
    x_val = np.stack((x_val[:, :, :, 0], x_val[:, :, :, 1], x_val[:, :, :, 2], x_val[:, :, :, 3]), axis = -1)
    x_test = np.stack((x_test[:, :, :, 0], x_test[:, :, :, 1], x_test[:, :, :, 2], x_test[:, :, :, 3]), axis = -1)
    '''
    featureNum = x_train.shape[3]
    # normalization
    xMin = np.min(x_train, axis = 0)
    xMax = np.max(x_train, axis = 0)
    x_train = (x_train - xMin) / (xMax - xMin)
    x_val = (x_val - xMin) / (xMax - xMin)
    x_test = (x_test - xMin) / (xMax - xMin)

    t1 = time.time()
    print("Finished data spliting in {:.3f}s".format(t1 - t0))
    t0 = time.time() 
    print("Setting up ...")
    inputs = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    inter = upStream(layers, shareConv, inputs)
    rr_branch = downStream(layers, shareConv, "rr", inter)
    tt_branch = downStream(layers, shareConv, "tt", inter)
    cos_branch = downStream(layers, shareConv, "cos", inter)
    sin_branch = downStream(layers, shareConv, "sin", inter)
    ze_branch = downStream(layers, shareConv, "ze", inter)
    cosAz_branch = downStream(layers, shareConv, "cosAz", inter)
    sinAz_branch = downStream(layers, shareConv, "sinAz", inter)
    sh_branch = downStream(layers, shareConv, "sh", inter)
    model = Model(inputs = inputs, outputs = [rr_branch, tt_branch, cos_branch, sin_branch, ze_branch, cosAz_branch, sinAz_branch, sh_branch])
    opt = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False)
    #model.compile(loss = {"rr_output":my_mspe, "tt_output":"mse", "cos_output":"mse", "sin_output":"mse", "ze_output":"mse", "cosAz_output":"mse", "sinAz_output":"mse", "sh_output":"mse"}, optimizer = opt, loss_weights = {"rr_output":100., "tt_output":10000., "cos_output":10000., "sin_output":10000., "ze_output":100., "cosAz_output":100., "sinAz_output":100., "sh_output":1.}) # vertex, nu direction, shower energies
    model.compile(loss = {"rr_output":my_mspe, "tt_output":"mse", "cos_output":"mse", "sin_output":"mse", "ze_output":"mse", "cosAz_output":"mse", "sinAz_output":"mse", "sh_output":"mse"}, optimizer = opt, loss_weights = {"rr_output":100., "tt_output":10000., "cos_output":10000., "sin_output":10000., "ze_output":0., "cosAz_output":0., "sinAz_output":0., "sh_output":0.}) # vertex only reconstruction
    
    # save the best weight set
    checkpoint = ModelCheckpoint("{}/allPairsWeights_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.hdf5".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor), save_best_only = True, verbose = 0, monitor = 'val_loss', mode = 'min')
    #model.summary()
    # plot architecture
    tf.keras.utils.plot_model(model, "{}/arch_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor), show_shapes = True)

    if Pred == "train":
        t1 = time.time()
        print("Finished setup in {:.3f}s".format(t1 - t0))
        t0 = time.time()
        print("Training ...")
        # dropout
        x_train[:, :, :, 0] = dropout(x_train[:, :, :, 0], rate = d0)
        x_train[:, :, :, 1] = dropout(x_train[:, :, :, 1], rate = d0)
        x_val[:, :, :, 0] = dropout(x_val[:, :, :, 0], rate = d0)
        x_val[:, :, :, 1] = dropout(x_val[:, :, :, 1], rate = d0)
        #history = model.fit(x_train, [y_train[:, 0], y_train[:, 1], y_train[:, 4], y_train[:, 5], y_train[:, 6], y_train[:, 8], y_train[:, 10], y_train[:, 11], y_train[:, 12]], epochs = epochs, batch_size = batch, verbose = 1, validation_data = (x_val, [y_val[:, 0], y_val[:, 1], y_val[:, 4], y_val[:, 5], y_val[:, 6], y_val[:, 8], y_val[:, 10], y_val[:, 11], y_val[:, 12]]), callbacks = [checkpoint, EarlyStopping(monitor = 'val_loss', patience = 8, verbose = 1, mode = 'min'), ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, verbose = 1, patience = 3)]) # with early stopping
        history = model.fit(x_train, [y_train[:, 0], y_train[:, 4], y_train[:, 5], y_train[:, 6], y_train[:, 8], y_train[:, 10], y_train[:, 11], y_train[:, 12]], epochs = epochs, batch_size = batch, verbose = 1, validation_data = (x_val, [y_val[:, 0], y_val[:, 4], y_val[:, 5], y_val[:, 6], y_val[:, 8], y_val[:, 10], y_val[:, 11], y_val[:, 12]]), callbacks = [checkpoint, ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, verbose = 0, patience = 3)]) # fixed epochs
        #print(history.history.keys())
        plotLearn("combine", history)
        plotLearn("total", history)
        plotLearn("rr", history)
        plotLearn("tt", history)
        plotLearn("cos", history)
        plotLearn("sin", history)
        plotLearn("ze", history)
        plotLearn("cosAz", history)
        plotLearn("sinAz", history)
        plotLearn("sh", history)
        #plotImportance(model, x_test, y_test)
        t1 = time.time()
        print("Finished training in {:.3f}s".format(t1 - t0))
        sys.exit("Trained successfully!") # finish train and exit
    else:
        break # use the intended fold for test and validation plots

t1 = time.time()
print("Finished setup in {:.3f}s".format(t1 - t0))
t0 = time.time()
print("Making predictions ...")
model.load_weights("{}/allPairsWeights_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn.hdf5".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor))
# make prediction
train_pred = np.array(model.predict(x_train, batch_size = batch))
test_pred = np.array(model.predict(x_test, batch_size = batch))
rr_train = y_train[:, 0].reshape((len(y_train), 1))
rr_test = y_test[:, 0].reshape((len(y_test), 1))
rr_train_pred = train_pred[0]
rr_test_pred = test_pred[0]
zz_train = y_train[:, 1].reshape((len(y_train), 1))
zz_test = y_test[:, 1].reshape((len(y_test), 1))
tt_train = np.degrees(y_train[:, 4].reshape((len(y_train), 1)))
tt_test = np.degrees(y_test[:, 4].reshape((len(y_test), 1)))
tt_test_pred = np.degrees(test_pred[1])
tt_train_pred = np.degrees(train_pred[1])
pp_train = np.degrees(y_train[:, 3].reshape((len(y_train), 1)))
pp_test = np.degrees(y_test[:, 3].reshape((len(y_test), 1)))
pp_test_pred = np.arctan2(test_pred[3], test_pred[2])
pp_test_pred = np.degrees(np.where(pp_test_pred < 0, pp_test_pred + 2 * np.pi, pp_test_pred))
pp_train_pred = np.arctan2(train_pred[3], train_pred[2])
pp_train_pred = np.degrees(np.where(pp_train_pred < 0, pp_train_pred + 2 * np.pi, pp_train_pred))
ze_train = np.degrees(y_train[:, 8].reshape((len(y_train), 1)))
ze_test = np.degrees(y_test[:, 8].reshape((len(y_test), 1)))
ze_test_pred = np.degrees(test_pred[4])
ze_train_pred = np.degrees(train_pred[4])
az_train = np.degrees(y_train[:, 7].reshape((len(y_train), 1)))
az_test = np.degrees(y_test[:, 7].reshape((len(y_test), 1)))
az_test_pred = np.arctan2(test_pred[6], test_pred[5])
az_test_pred = np.degrees(np.where(az_test_pred < 0, az_test_pred + 2 * np.pi, az_test_pred))
az_train_pred = np.arctan2(train_pred[6], train_pred[5])
az_train_pred = np.degrees(np.where(az_train_pred < 0, az_train_pred + 2 * np.pi, az_train_pred))
sh_train = y_train[:, 12].reshape((len(y_train), 1))
sh_test = y_test[:, 12].reshape((len(y_test), 1))
sh_test_pred = test_pred[7]
sh_train_pred = train_pred[7]
en_train = y_train[:, 9].reshape((len(y_train), 1))
en_test = y_test[:, 9].reshape((len(y_test), 1))
xx_train = y_train[:, 13].reshape((len(y_train), 1))
xx_test = y_test[:, 13].reshape((len(y_test), 1))
yy_train = y_train[:, 14].reshape((len(y_train), 1))
yy_test = y_test[:, 14].reshape((len(y_test), 1))
fl_train = y_train[:, 15].reshape((len(y_train), 1))
fl_test = y_test[:, 15].reshape((len(y_test), 1))
viewAngle_dir_train = np.degrees(y_train[:, 16].reshape((len(y_train), 1)))
viewAngle_dir_test = np.degrees(y_test[:, 16].reshape((len(y_test), 1)))
viewAngle_ref_train = np.degrees(y_train[:, 17].reshape((len(y_train), 1)))
viewAngle_ref_test = np.degrees(y_test[:, 17].reshape((len(y_test), 1)))
coneAngle_dir_train = np.degrees(y_train[:, 18].reshape((len(y_train), 1)))
coneAngle_dir_test = np.degrees(y_test[:, 18].reshape((len(y_test), 1)))
coneAngle_ref_train = np.degrees(y_train[:, 19].reshape((len(y_train), 1)))
coneAngle_ref_test = np.degrees(y_test[:, 19].reshape((len(y_test), 1)))
SNRs_train = np.log10(y_train[:, 20].reshape((len(y_train), 1)))
SNRs_test = np.log10(y_test[:, 20].reshape((len(y_test), 1)))
evt_test = y_test[:, 21].reshape((len(y_test), 1))
if inMode == 1:
    unix_test = y_test[:, 22].reshape((len(y_test), 1))
quatile = [0., np.nanpercentile(SNRs_test, 25), np.nanpercentile(SNRs_test, 50), np.nanpercentile(SNRs_test, 75), float("inf")]
if Pred == "pp":
    y_test_pred = pp_test_pred
    y_train_pred = pp_train_pred
    y_test = pp_test
    y_train = pp_train
elif Pred == "az":
    y_test_pred = az_test_pred
    y_train_pred = az_train_pred
    y_test = az_test
    y_train = az_train
elif Pred == "rr":
    y_test_pred = rr_test_pred
    y_train_pred = rr_train_pred
    y_test = rr_test
    y_train = rr_train
elif Pred == "tt":
    y_test_pred = tt_test_pred
    y_train_pred = tt_train_pred
    y_test = tt_test
    y_train = tt_train
elif Pred == "ze":
    y_test_pred = ze_test_pred
    y_train_pred = ze_train_pred
    y_test = ze_test
    y_train = ze_train
elif Pred == "sh":
    y_test_pred = sh_test_pred
    y_train_pred = sh_train_pred
    y_test = sh_test
    y_train = sh_train

t1 = time.time()
print("Finished prediction in {:.3f}s".format(t1 - t0))
t0 = time.time()
print("Making plots ...")
meanPerEnergies = []
sdPerEnergies = []
meanPerEnergiesNoout = []
sdPerEnergiesNoout = []
me = 0
sd = 0
meNoout = 0
sdNoout = 0
medNooutPerQ = []
sigLNooutPerQ = []
sigRNooutPerQ = []
diff_test = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
diff_train = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
# consider the boundry case at 0/360 deg
if Pred == "pp" or Pred == "az":
    diff_test = np.where(diff_test > 180., diff_test - 360., diff_test)
    diff_test = np.where(diff_test < -180., diff_test + 360., diff_test)
    diff_train = np.where(diff_train > 180., diff_train - 360., diff_train)
    diff_train = np.where(diff_train < -180., diff_train + 360., diff_train)
# find outliers
if Pred == "rr":
    outliers = np.where(np.abs(diff_test / y_test.reshape((len(y_test), 1))) > 0.5, True, False)
elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    outliers = np.where(np.abs(diff_test) > 20., True, False)
elif Pred == "tt" or Pred == "pp" or Pred == "en" or Pred == "sh":
    outliers = np.where(np.abs(diff_test) > 2., True, False)

#mean error 1dhist per energy
plt.rc('font', size = 5)
for i in range(len(Energies)):
    outSelect = outliers[en_test == Energies[i]]
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    if Pred == "rr":
        sdPerEnergies.append(np.std(diff_test[en_test == Energies[i]] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))))
        meanPerEnergies.append(np.mean(diff_test[en_test == Energies[i]] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))))
        sdPerEnergiesNoout.append(np.std(diff_test[en_test == Energies[i]][~outSelect] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))[~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff_test[en_test == Energies[i]][~outSelect] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))[~outSelect]))
        #plt.hist(diff_test[en_test == Energies[i]] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1)), bins = 200, range = (-0.5, 0.51), density = True)
        plt.hist(diff_test[en_test == Energies[i]] / y_test[en_test == Energies[i]], bins = 200, range = (-0.5, 0.51), density = True)
        plt.xlim((-0.5, 0.5))
        plt.ylim((0, 6))
        plt.xlabel("{}_relativeError".format(Pred))
    elif Pred == "tt" or Pred == "pp":
        sdPerEnergies.append(np.std(diff_test[en_test == Energies[i]]))
        meanPerEnergies.append(np.mean(diff_test[en_test == Energies[i]]))
        sdPerEnergiesNoout.append(np.std(diff_test[en_test == Energies[i]][~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff_test[en_test == Energies[i]][~outSelect]))
        plt.hist(diff_test[en_test == Energies[i]], bins = 200, range = (-2.0, 2.01), density = True)
        plt.xlim((-2, 2))
        plt.ylim((0, 3))
        plt.xlabel("{}_error[deg]".format(Pred))
    elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        sdPerEnergies.append(np.std(diff_test[en_test == Energies[i]]))
        meanPerEnergies.append(np.mean(diff_test[en_test == Energies[i]]))
        sdPerEnergiesNoout.append(np.std(diff_test[en_test == Energies[i]][~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff_test[en_test == Energies[i]][~outSelect]))
        plt.hist(diff_test[en_test == Energies[i]], bins = 200, range = (-20.0, 20.01), density = True)
        plt.xlim((-20, 20))
        plt.ylim((0, 0.4))
        plt.xlabel("{}_error[deg]".format(Pred))
    elif Pred == "sh":
        sdPerEnergies.append(np.std(diff_test[en_test == Energies[i]]))
        meanPerEnergies.append(np.mean(diff_test[en_test == Energies[i]]))
        sdPerEnergiesNoout.append(np.std(diff_test[en_test == Energies[i]][~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff_test[en_test == Energies[i]][~outSelect]))
        plt.hist(diff_test[en_test == Energies[i]], bins = 200, range = (-2.0, 2.01), density = True)
        plt.xlim((-2, 2))
        plt.ylim((0, 3))
        plt.xlabel("{}_error[log10(eV)]".format(Pred))
    plt.ylabel("{}_probability".format(Pred))
    plt.grid(True)
    plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sd = {:.3f}, {}_mean = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
plt.tight_layout()
plt.savefig("{}/meanErrorPerEnergy_1dHist_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanErrorPerEnergy_1dHist_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#mean error 1dhist quatile by quatile
plt.rc('font', size = 10)
for i in range(4):
    tmpN = diff_test[(SNRs_test > quatile[i]) & (SNRs_test < quatile[i + 1]) & (~outliers)]
    tmpD = y_test[(SNRs_test > quatile[i]) & (SNRs_test < quatile[i + 1]) & (~outliers)]
    plt.subplot(2, 2, i + 1)
    if Pred == "rr":
        meNoout = np.mean(tmpN / tmpD)
        sdNoout = np.std(tmpN / tmpD)
        medNooutPerQ.append(np.nanpercentile(tmpN / tmpD, 50))
        sigLNooutPerQ.append(np.nanpercentile(tmpN / tmpD, 16))
        sigRNooutPerQ.append(np.nanpercentile(tmpN / tmpD, 84))
        plt.hist(tmpN / tmpD, bins = 200, range = (-0.5, 0.51), density = True)
        plt.xlim((-0.5, 0.5))
        plt.ylim((0, 8))
        plt.xlabel("{}_relativeError".format(Pred))
        RE = np.linspace(-1, 1, 200)
        plt.plot(RE, stats.norm.pdf(RE, meNoout, sdNoout), linewidth = 1, color = "r")
        plt.plot(RE, stats.norm.pdf(RE, medNooutPerQ[i], (sigRNooutPerQ[i] - sigLNooutPerQ[i]) / 2.), linewidth = 1, color = "b")
    elif Pred == "tt" or Pred == "pp":
        meNoout = np.mean(tmpN)
        sdNoout = np.std(tmpN)
        medNooutPerQ.append(np.nanpercentile(tmpN, 50))
        sigLNooutPerQ.append(np.nanpercentile(tmpN, 16))
        sigRNooutPerQ.append(np.nanpercentile(tmpN, 84))
        plt.hist(tmpN, bins = 200, range = (-2.0, 2.01), density = True)
        plt.xlim((-2, 2))
        plt.ylim((0, 3))
        plt.xlabel("{}_error[deg]".format(Pred))
        RE = np.linspace(-5, 5, 200)
        plt.plot(RE, stats.norm.pdf(RE, meNoout, sdNoout), linewidth = 1, color = "r")
        plt.plot(RE, stats.norm.pdf(RE, medNooutPerQ[i], (sigRNooutPerQ[i] - sigLNooutPerQ[i]) / 2.), linewidth = 1, color = "b")
    elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        meNoout = np.mean(tmpN)
        sdNoout = np.std(tmpN)
        medNooutPerQ.append(np.nanpercentile(tmpN, 50))
        sigLNooutPerQ.append(np.nanpercentile(tmpN, 16))
        sigRNooutPerQ.append(np.nanpercentile(tmpN, 84))
        plt.hist(tmpN, bins = 200, range = (-20.0, 20.01), density = True)
        plt.xlim((-20, 20))
        plt.ylim((0, 0.4))
        plt.xlabel("{}_error[deg]".format(Pred))
        RE = np.linspace(-20, 20, 200)
        plt.plot(RE, stats.norm.pdf(RE, meNoout, sdNoout), linewidth = 1, color = "r")
        plt.plot(RE, stats.norm.pdf(RE, medNooutPerQ[i], (sigRNooutPerQ[i] - sigLNooutPerQ[i]) / 2.), linewidth = 1, color = "b")
    elif Pred == "sh":
        meNoout = np.mean(tmpN)
        sdNoout = np.std(tmpN)
        medNooutPerQ.append(np.nanpercentile(tmpN, 50))
        sigLNooutPerQ.append(np.nanpercentile(tmpN, 16))
        sigRNooutPerQ.append(np.nanpercentile(tmpN, 84))
        plt.hist(tmpN, bins = 200, range = (-2.0, 2.01), density = True)
        plt.xlim((-2, 2))
        plt.ylim((0, 3))
        plt.xlabel("{}_error[log10(eV)]".format(Pred))
        RE = np.linspace(-5, 5, 200)
        plt.plot(RE, stats.norm.pdf(RE, meNoout, sdNoout), linewidth = 1, color = "r")
        plt.plot(RE, stats.norm.pdf(RE, medNooutPerQ[i], (sigRNooutPerQ[i] - sigLNooutPerQ[i]) / 2.), linewidth = 1, color = "b")
    plt.ylabel("{}_probability".format(Pred))
    plt.grid(True)
    plt.title("quatile {}\n{} samples\nsd = {:.3f}, mean = {:.3f}\nsigL = {:.3f}, med = {:.3f}, sigR = {:.3f}".format(i + 1, len(tmpN), sdNoout, meNoout, sigLNooutPerQ[i], medNooutPerQ[i], sigRNooutPerQ[i]))
plt.tight_layout()
plt.savefig("{}/meanErrorQuatile_1dHist_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanErrorQuatile_1dHist_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#mean error 1dhist all
plt.rc('font', size = 16)
ax = plt.gca()
tmp = stats.norm.pdf(RE, medNooutPerQ[0], (sigRNooutPerQ[0] - sigLNooutPerQ[0]) / 2.)
for i in range(1, 4):
    tmp += stats.norm.pdf(RE, medNooutPerQ[i], (sigRNooutPerQ[i] - sigLNooutPerQ[i]) / 2.)
tmp /= 4.
if Pred == "rr":
    me = np.mean(diff_test / y_test.reshape((len(y_test), 1)))
    sd = np.std(diff_test / y_test.reshape((len(y_test), 1)))
    meNoout = np.mean(diff_test[~outliers] / y_test.reshape((len(y_test), 1))[~outliers])
    sdNoout = np.std(diff_test[~outliers] / y_test.reshape((len(y_test), 1))[~outliers])
    medNoout = np.nanpercentile(diff_test[~outliers] / y_test.reshape((len(y_test), 1))[~outliers], 50)
    sigLNoout = np.nanpercentile(diff_test[~outliers] / y_test.reshape((len(y_test), 1))[~outliers], 16)
    sigRNoout = np.nanpercentile(diff_test[~outliers] / y_test.reshape((len(y_test), 1))[~outliers], 84)
    plt.hist(diff_test / y_test.reshape((len(y_test), 1)), bins = 200, range = (-0.5, 0.51), density = True)
    plt.text(0.0, 0.8, "16 percentile = {:.3f}\nmedian = {:.3f}\n84 percentile = {:.3f}".format(sigLNoout, medNoout, sigRNoout), transform = ax.transAxes)
    plt.xlim((-0.5, 0.5))
    plt.ylim((0, 10))
    plt.xlabel("$\\Delta$r/r")
    RE = np.linspace(-1, 1, 200)
elif Pred == "tt" or Pred == "pp":
    me = np.mean(diff_test)
    sd = np.std(diff_test)
    meNoout = np.mean(diff_test[~outliers])
    sdNoout = np.std(diff_test[~outliers])
    medNoout = np.nanpercentile(diff_test[~outliers], 50)
    sigLNoout = np.nanpercentile(diff_test[~outliers], 16)
    sigRNoout = np.nanpercentile(diff_test[~outliers], 84)
    plt.hist(diff_test, bins = 200, range = (-2.0, 2.01), density = True)
    plt.text(0.0, 0.8, "16 percentile = {:.3f}\nmedian = {:.3f}\n84 percentile = {:.3f}".format(sigLNoout, medNoout, sigRNoout), transform = ax.transAxes)
    plt.xlim((-2, 2))
    plt.ylim((0, 3))
    if Pred == "tt":
        plt.xlabel("$\\Delta\\beta$[deg]")
    elif Pred == "pp":
        plt.xlabel("$\\Delta\\alpha$[deg]")
    RE = np.linspace(-5, 5, 200)
elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    me = np.mean(diff_test)
    sd = np.std(diff_test)
    meNoout = np.mean(diff_test[~outliers])
    sdNoout = np.std(diff_test[~outliers])
    medNoout = np.nanpercentile(diff_test[~outliers], 50)
    sigLNoout = np.nanpercentile(diff_test[~outliers], 16)
    sigRNoout = np.nanpercentile(diff_test[~outliers], 84)
    plt.hist(diff_test, bins = 200, range = (-20.0, 20.01), density = True)
    plt.text(0.0, 0.8, "16 percentile = {:.3f}\nmedian = {:.3f}\n84 percentile = {:.3f}".format(sigLNoout, medNoout, sigRNoout), transform = ax.transAxes)
    plt.xlim((-20, 20))
    plt.ylim((0, 0.2))
    if Pred == "ze":
        plt.xlabel("$\\Delta\\theta$[deg]")
    elif Pred == "az":
        plt.xlabel("$\\Delta\\phi$[deg]")
    RE = np.linspace(-20, 20, 200)
elif Pred == "sh":
    me = np.mean(diff_test)
    sd = np.std(diff_test)
    meNoout = np.mean(diff_test[~outliers])
    sdNoout = np.std(diff_test[~outliers])
    medNoout = np.nanpercentile(diff_test[~outliers], 50)
    sigLNoout = np.nanpercentile(diff_test[~outliers], 16)
    sigRNoout = np.nanpercentile(diff_test[~outliers], 84)
    plt.hist(diff_test, bins = 200, range = (-2.0, 2.01), density = True)
    plt.text(0.0, 0.8, "16 percentile = {:.3f}\nmedian = {:.3f}\n84 percentile = {:.3f}".format(sigLNoout, medNoout, sigRNoout), transform = ax.transAxes)
    plt.xlim((-2, 2))
    plt.ylim((0, 3))
    plt.xlabel("{}_error[log10(eV)]".format(Pred))
    RE = np.linspace(-5, 5, 200)
plt.ylabel("probability density".format(Pred))
#plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_1dHist_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_1dHist_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#outliers rr vs zz plots
plt.rc('font', size = 5)
for i in range(len(Energies)):
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    outSelect = outliers[en_test == Energies[i]]
    plt.scatter(rr_test[en_test == Energies[i]][outSelect], -1.0 * zz_test[en_test == Energies[i]][outSelect], s = 1)
    plt.xlabel("{}_rr[m]".format(Pred))
    plt.ylabel("{}_zz[m]".format(Pred))
    plt.xlim((0, 8000))
    plt.ylim((-3000, 0))
    plt.title("10^{:.3f} eV".format(Energies[i]))
    plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("{}/outliers_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/outliers_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#true vs pred
plt.rc('font', size = 5)
for i in range(len(Energies)):
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    if y_test[en_test == Energies[i]].shape[0] == 0:
        continue
    if Pred == "rr":
        plt.plot([0, 8000], [0, 8000], c = "r", linewidth = 0.5)
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), y_test_pred[en_test == Energies[i]].reshape((len(y_test_pred[en_test == Energies[i]]),)), range = [[0, 8000], [0, 8000]], bins=[50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.xlabel("{}_test[m]".format(Pred))
        plt.ylabel("{}_pred[m]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdRelativeError = {:.3f}\n{}_meanRelativeError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "en" or Pred == "sh":
        plt.plot([15, 21], [15, 21], c = "r", linewidth = 0.5)
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), y_test_pred[en_test == Energies[i]].reshape((len(y_test_pred[en_test == Energies[i]]),)), range = [[15, 21], [15, 21]], bins=[50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.xlabel("{}_test[log10(eV)]".format(Pred))
        plt.ylabel("{}_pred[log10(eV)]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "pp" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        plt.plot([0, 360], [0, 360], c = "r", linewidth = 0.5)
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), y_test_pred[en_test == Energies[i]].reshape((len(y_test_pred[en_test == Energies[i]]),)), range = [[0, 360], [0, 360]], bins=[50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylabel("{}_pred[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "tt":
        plt.plot([-90, 10], [-90, 10], c = "r", linewidth = 0.5)
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), y_test_pred[en_test == Energies[i]].reshape((len(y_test_pred[en_test == Energies[i]]),)), range = [[-90, 10], [-90, 10]], bins = [50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylabel("{}_pred[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
plt.tight_layout()
plt.savefig("{}/trueVsPred_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/trueVsPred_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

# diff vs true
plt.rc('font', size = 5)
for i in range(len(Energies)):
    diff = np.array(y_test[en_test == Energies[i]] - y_test_pred[en_test == Energies[i]])
    diff = diff.reshape((diff.shape[0],))
    if Pred == "pp" or Pred == "az":
        diff = np.where(diff > 180., diff - 360., diff)
        diff = np.where(diff < -180., diff + 360., diff)
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    if y_test[en_test == Energies[i]].shape[0] == 0:
        continue
    if Pred == "rr":
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), diff / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), range = [[0, 8000], [-1, 1]], bins = [50, 50], cmap = plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.ylabel("{}_relative error".format(Pred))
        plt.xlabel("{}_test[m]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdRelativeError = {:.3f}\n{}_meanRelativeError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "pp":
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), diff, range = [[0, 360], [-2, 2]], bins = [50, 50], cmap = plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "tt":
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), diff, range = [[-90, 10], [-2, 2]], bins = [50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), diff, range = [[0, 360], [-20, 20]], bins = [50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    elif Pred == "en" or Pred == "sh":
        plt.hist2d(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]),)), diff, range = [[15, 21], [-2, 2]], bins=[50, 50], cmap=plt.get_cmap('Blues'))
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.ylabel("{}_error[log10(eV)]".format(Pred))
        plt.xlabel("{}_test[log10(eV)]".format(Pred))
        plt.ylim((-2.0, 2.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i]))
    plt.grid(True)
plt.tight_layout()
plt.savefig("{}/trueVsDiff_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/trueVsDiff_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

# diff vs snr
plt.rc('font', size = 10)
if Pred == "rr":
    plt.hist2d(SNRs_test.reshape((len(SNRs_test),)), diff_test.reshape((diff_test.shape[0],)) / y_test.reshape((len(y_test),)), range = [[0, 3], [-1, 1]], bins = [50, 50], cmap = plt.get_cmap('Blues'))
    cb = plt.colorbar()
    cb.set_label("counts")
    plt.ylabel("{}_relative error".format(Pred))
elif Pred == "pp" or Pred == "tt":
    plt.hist2d(SNRs_test.reshape((len(SNRs_test),)), diff_test.reshape((diff_test.shape[0],)), range = [[0, 3], [-2, 2]], bins = [50, 50], cmap = plt.get_cmap('Blues'))
    cb = plt.colorbar()
    cb.set_label("counts")
    plt.ylabel("{}_error[deg]".format(Pred))
elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    plt.hist2d(SNRs_test.reshape((len(SNRs_test),)), diff_test.reshape((diff_test.shape[0],)), range = [[0, 3], [-20, 20]], bins = [50, 50], cmap=plt.get_cmap('Blues'))
    cb = plt.colorbar()
    cb.set_label("counts")
    plt.ylabel("{}_error[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    plt.hist2d(SNRs_test.reshape((len(SNRs_test),)), diff_test.reshape((diff_test.shape[0],)), range = [[0, 3], [-2, 2]], bins=[50, 50], cmap=plt.get_cmap('Blues'))
    cb = plt.colorbar()
    cb.set_label("counts")
    plt.ylabel("{}_error[log10(eV)]".format(Pred))
plt.vlines([quatile[1], quatile[2], quatile[3]], ymin = -20, ymax = 20, colors = "r")
plt.xlabel("{}_log10(snr)".format(Pred))
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/diffVsSNR_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/diffVsSNR_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf() 

#2dhist of mean error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_rzHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_rzHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#2dhist of mean error for training set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_rzHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_rzHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#xy 2dhist of mean error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_xyHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_xyHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#xy 2dhist of mean error for train set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_xyHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_xyHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#direction 2dhist of mean error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_dirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_dirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#direction 2dhist of mean error for train set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_dirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_dirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of mean error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.ylabel("viewing angle[deg]")
plt.xlabel("Cone Angle Dir[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_viewVsConeDirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_viewVsConeDirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of mean error for train set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.ylabel("viewing angle[deg]")
plt.xlabel("Cone Angle Dir[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_viewVsConeDirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_viewVsConeDirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of std error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_dir_test.reshape((len(coneAngle_dir_test),)), viewAngle_dir_test.reshape((len(viewAngle_dir_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.ylabel("viewing angle[deg]")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Dir[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_viewVsConeDirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_viewVsConeDirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of std error for train set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.ylabel("viewing angle[deg]")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Dir[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_viewVsConeDirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_viewVsConeDirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of mean error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.ylabel("viewing angle[deg]")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Ref[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_viewVsConeRefHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_viewVsConeRefHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of mean error for train set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -20., vmax = 20., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "mean", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanError[log10(eV)]".format(Pred))
plt.ylabel("viewing angle[deg]")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Ref[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/meanError_viewVsConeRefHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/meanError_viewVsConeRefHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of std error for test set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_ref_test.reshape((len(coneAngle_ref_test),)), viewAngle_ref_test.reshape((len(viewAngle_ref_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.ylabel("viewing angle[deg]")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Ref[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_viewVsConeRefHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_viewVsConeRefHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of std error for train set linear
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-180, 180, 5), np.arange(0, 110, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 110, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.ylabel("viewing angle[deg]")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Ref[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_viewVsConeRefHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_viewVsConeRefHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#xy 2dhist of std error for test set
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_xyHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_xyHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#xy 2dhist of std error for train set
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200)])
    plot_x, plot_y = np.meshgrid(np.arange(-8000, 8000, 200), np.arange(-8000, 8000, 200))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_xyHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_xyHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#direction 2dhist of std error for test set
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_dirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_dirHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#direction 2dhist of std error for train set
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 360, 5), np.arange(0, 180, 5)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 360, 5), np.arange(0, 180, 5))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_dirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_dirHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#2dhist of std error linear for test set
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., (diff_test / y_test.reshape((y_test.shape[0], 1))).reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., diff_test.reshape((diff_test.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_rzHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_rzHistLinear_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#2dhist of std error linear for train set
plt.rc('font', size = 10)
if Pred == "rr":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., (diff_train / y_train.reshape((y_train.shape[0], 1))).reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdRelativeError".format(Pred))
elif Pred == "pp" or Pred == "tt":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 20, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[deg]".format(Pred))
elif Pred == "en" or Pred == "sh":
    hist = stats.binned_statistic_2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., diff_train.reshape((diff_train.shape[0],)), statistic = "std", bins = [np.arange(0, 8000, 100), np.arange(-3000, 0, 100)])
    plot_x, plot_y = np.meshgrid(np.arange(0, 8000, 100), np.arange(-3000, 0, 100))
    plt.pcolormesh(plot_x, plot_y, hist[0].T, vmin = 0., vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_stdError[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("{}/stdError_rzHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/stdError_rzHistLinear_train_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

#error summary
plt.rc('font', size = 10)
plt.subplot(2, 1, 1)
plt.grid(True)
plt.xlabel('Energies[log10(eV)]')
plt.plot(Energies, sdPerEnergies, marker = "x", label = "sdPerEnergies")
plt.plot(Energies, sdPerEnergiesNoout, marker = "o", label = "sdPerEnergiesNoout")
if Pred == "rr":
    plt.ylabel('{}_sdRelativeError'.format(Pred))
    plt.ylim(0., 0.5)
elif Pred == "pp" or Pred == "tt":
    plt.ylabel('{}_sdError[deg]'.format(Pred))
    plt.ylim(0., 2.)
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    plt.ylabel('{}_sdError[deg]'.format(Pred))
    plt.ylim(0., 20.)
elif Pred == "sh":
    plt.ylabel('{}_sdError[log10(eV)]'.format(Pred))
    plt.ylim(0., 2.)
plt.legend()
plt.subplot(2, 1, 2)
plt.grid(True)
plt.xlabel('Energies[log10(eV)]')
plt.plot(Energies, meanPerEnergies, marker = "x", label = "meanPerEnergies")
plt.plot(Energies, meanPerEnergiesNoout, marker = "o", label = "meanPerEnergiesNoout")
if Pred == "rr":
    plt.ylabel('{}_meanRelativeError'.format(Pred))
    plt.ylim(-0.5, 0.5)
elif Pred == "pp" or Pred == "tt":
    plt.ylabel('{}_meanError[deg]'.format(Pred))
    plt.ylim(-2, 2)
elif Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    plt.ylabel('{}_meanError[deg]'.format(Pred))
    plt.ylim(-20, 20)
elif Pred == "sh":
    plt.ylabel('{}_meanError[log10(eV)]'.format(Pred))
    plt.ylim(-2, 2)
plt.legend()
plt.tight_layout()
plt.savefig("{}/sum_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
print("saving {}/sum_test_{}layers{}nodes{}epochs{}batch{}fold{}an{}tn_{}.pdf".format(outPath, layers, nodes, epochs, batch, fold, ampNoiseFactor, timeNoiseFactor, Pred))
plt.clf()

t1 = time.time()
print("Finished plotting in {:.3f}s".format(t1 - t0))
t0 = time.time()
