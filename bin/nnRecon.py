print("Importing packages ...")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats as stats
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
from scipy.stats import lognorm, norm, sem
from NuRadioMC.utilities import medium
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import pandas as pd 
import seaborn as sbn
import sklearn
import keras
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Input, MaxPooling3D, Conv3D, MaxPooling2D, Conv2D, Masking, Embedding
from keras import backend as K
from radiotools import helper as hp

def my_mspe(y_true, y_pred):
    # self defined mean squared percentage error
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.square((y_true - y_pred) / K.maximum(math_ops.abs(y_true), K.epsilon()))
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
    deno = np.maximum(maximum - minimum, 1e-10)# avoid dividing by zero
    mat = (mat - minimum) / deno
    return mat.reshape((mat.shape[0], nRow, nCol))

def share(layers, inputs):
    # the shared layers of different branches
    x = Conv2D(nodes, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "elu")(inputs)
    for i in range(layers - 1):
        x = Conv2D(nodes, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "elu")(x)
    x = Flatten()(x)
    return x
    
def separate(layers, pred, inputs):
    # the separate layers of different branches
    x = Conv2D(nodes, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "elu")(inputs)
    for i in range(layers - 1):
        x = Conv2D(nodes, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "elu")(x)
    x = Flatten()(x)
    for i in range(layers):
        x = Dense(nodes, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation='elu')(x)
    x = Dense(1, name = "{}_output".format(pred))(x)
    return x

def plotFeature(x, y):
    # plot relationships between different feature pairs
    data = {"t_dir[1, 0]": x[:, 1, 0, 0],
            "t_ref[1, 0]": x[:, 1, 0, 1],
            "a_dir[1, 0]": x[:, 1, 0, 2],
            "a_ref[1, 0]": x[:, 1, 0, 3],
            "r_dir[1, 0]": x[:, 1, 0, 4],
            "r_ref[1, 0]": x[:, 1, 0, 5],
            "v_dir[1, 0]": x[:, 1, 0, 6],
            "v_ref[1, 0]": x[:, 1, 0, 7],
            "c_dir[1, 0]": x[:, 1, 0, 8],
            "c_ref[1, 0]": x[:, 1, 0, 9],
            "t_dir[2, 0]": x[:, 2, 0, 0],
            "t_ref[2, 0]": x[:, 2, 0, 1],
            "a_dir[2, 0]": x[:, 2, 0, 2],
            "a_ref[2, 0]": x[:, 2, 0, 3],
            "r_dir[2, 0]": x[:, 2, 0, 4],
            "r_ref[2, 0]": x[:, 2, 0, 5],
            "v_dir[2, 0]": x[:, 2, 0, 6],
            "v_ref[2, 0]": x[:, 2, 0, 7],
            "c_dir[2, 0]": x[:, 2, 0, 8],
            "c_ref[2, 0]": x[:, 2, 0, 9],
            "t_dir[0, 1]": x[:, 0, 1, 0],
            "t_ref[0, 1]": x[:, 0, 1, 1],
            "a_dir[0, 1]": x[:, 0, 1, 2],
            "a_ref[0, 1]": x[:, 0, 1, 3],
            "r_dir[0, 1]": x[:, 0, 1, 4],
            "r_ref[0, 1]": x[:, 0, 1, 5],
            "v_dir[0, 1]": x[:, 0, 1, 6],
            "v_ref[0, 1]": x[:, 0, 1, 7],
            "c_dir[0, 1]": x[:, 0, 1, 8],
            "c_ref[0, 1]": x[:, 0, 1, 9],
            "rr": y[:, 0],
            "zz": y[:, 1],
            "pp": y[:, 3],
            "tt": y[:, 4],
            "az": y[:, 7],
            "ze": y[:, 8],
            "sh": y[:, 12]}
    df = pd.DataFrame(data, columns = ["t_dir[1, 0]", "t_ref[1, 0]", "a_dir[1, 0]", "a_ref[1, 0]", "r_dir[1, 0]", "r_ref[1, 0]", "v_dir[1, 0]", "v_ref[1, 0]", "c_dir[1, 0]", "c_ref[1, 0]", "t_dir[2, 0]", "t_ref[2, 0]", "a_dir[2, 0]", "a_ref[2, 0]", "r_dir[2, 0]", "r_ref[2, 0]", "v_dir[2, 0]", "v_ref[2, 0]", "c_dir[2, 0]", "c_ref[2, 0]", "t_dir[0, 1]", "t_ref[0, 1]", "a_dir[0, 1]", "a_ref[0, 1]", "r_dir[0, 1]", "r_ref[0, 1]", "v_dir[0, 1]", "v_ref[0, 1]", "c_dir[0, 1]", "c_ref[0, 1]", "rr", "zz", "pp", "tt", "az", "ze", "sh"])
    
    pd.plotting.scatter_matrix(df[:200], figsize = (60, 60), alpha = 0.6, diagonal = "hist")
    plt.savefig("./plots/nnRecon/scat_{}layers{}nodes{}epochs{}batch{}fold.pdf".format(layers, nodes, epochs, batch, fold))
    plt.clf()
    
    plt.subplots(figsize = (6.4, 4.8))
    corrMatrix = df.corr()
    sbn.heatmap(corrMatrix, vmin = -1., vmax = 1., cmap = cm.RdBu)
    plt.tight_layout()
    plt.savefig("./plots/nnRecon/corr_{}layers{}nodes{}epochs{}batch{}fold.pdf".format(layers, nodes, epochs, batch, fold))
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
    plt.title("train on {} samples\nvalidate on {} samples\n{}layers{}nodes{}epochs{}batch{}fold".format(len(y_train), len(y_val), layers, nodes, epochs, batch, fold))        
    plt.legend()
    plt.grid(True)
    plt.xlabel("epochs")
    plt.ylabel("{}_loss".format(pred))
    plt.tight_layout()
    plt.savefig("./plots/nnRecon/loss_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, pred))
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
                origin = x[:, k, j, i].copy()
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
                x[:, k, j, i] = origin.copy()
    importance = np.array(importance)
    sbn.heatmap(importance, norm = LogNorm(vmin = 0.01, vmax = 100), linewidths = .5, linecolor = "black", cmap = cm.Blues)
    xlabel = ["total", "rr", "tt", "cos", "sin", "ze", "cosAz", "sinAz", "sh", "pp", "az", "az - pp"]
    plt.xticks(range(12), xlabel, rotation = "vertical")
    # in case of different input structures
    if x.shape[3] == 3:
        ylabel = ["t_dir", "a_dir", "r_dir"]
    elif x.shape[3] == 4:
        ylabel = ["t_dir", "t_ref", "r_dir", "r_ref"]
    elif x.shape[3] == 6:
        ylabel = ["t_dir", "t_ref", "a_dir", "a_ref", "r_dir", "r_ref"]
    plt.yticks(range(0, x.shape[1] * x.shape[2] * x.shape[3], x.shape[1] * x.shape[2]), ylabel, rotation = "vertical")
    plt.tight_layout()
    plt.savefig("./plots/nnRecon/importance_{}layers{}nodes{}epochs{}batch{}fold.pdf".format(layers, nodes, epochs, batch, fold))
    plt.clf()

numSplit = 10 # 10 fold cross validation bu default, 90% training, 5% validation, 5% test
strNum = 4 # number of strings for a detector
channelPerStr = 4 # number of antennas on each string
Energies = []
fold = int(sys.argv[-1]) # the foldth fold as validation and test
if fold > numSplit - 1:
    print("Input error! No \"{}th\" fold in {} fold xvalidation!".format(fold, numSplit))
    sys.exit(1)
batch = int(sys.argv[-2]) # batch size
epochs = int(sys.argv[-3]) # number of epochs for training
nodes = int(sys.argv[-4]) # number of nodes for each layer, constant for all the layers
layers = int(sys.argv[-5]) # number of layers for convolutional and fully connected parts, this is number for each part
print("Architecture: {} layers, {} nodes, {} epochs, {} batch, {} fold".format(layers, nodes, epochs, batch, fold))
Pred = sys.argv[-6] # flag for functionality
if Pred not in ["train", "rr", "tt", "pp", "ze", "az", "sh"]:
    print("Input error! No argument \"{}\"!".format(Pred))
    sys.exit(1)
print("Modeling: {}".format(Pred))
inFile = h5py.File(sys.argv[1], 'r') # start to read in files
print("Reading " + str(sys.argv[1]))
event_ids = np.array(inFile['event_ids']) + 1 * 10 ** 5
energies = np.array(inFile['energies'])
if np.round(np.log10(inFile['energies'][0]), 1) not in Energies:
    Energies.append(np.round(np.log10(inFile['energies'][0]), 1))
xx = np.array(inFile['xx'])
yy = np.array(inFile['yy'])
zz = np.array(inFile['zz'])
flavors = np.array(inFile['flavors'])
inelasticity = np.array(inFile['inelasticity'])
interaction_type = np.array(inFile['interaction_type'])
azimuths = np.array(inFile['azimuths'])
zeniths = np.array(inFile['zeniths'])
SNRs = np.array(inFile['station_101']['SNRs'])
max_amp_ray_solution = np.array(inFile['station_101']['max_amp_ray_solution'])
ray_tracing_solution_type = np.array(inFile['station_101']['ray_tracing_solution_type'])
travel_times = np.array(inFile['station_101']['travel_times'])
travel_distances = np.array(inFile['station_101']['travel_distances'])
receive_vectors = np.array(inFile['station_101']['receive_vectors'])
launch_vectors = np.array(inFile['station_101']['launch_vectors'])
polarization = np.array(inFile['station_101']['polarization'])
for i in range(2, len(sys.argv) - 6):
    inFile = h5py.File(sys.argv[i], 'r')
    print("Reading " + str(sys.argv[i]))
    event_ids = np.append(event_ids, np.array(inFile['event_ids']) + ((i - 2) / 8 + 1) * 10**5)
    energies = np.append(energies, np.array(inFile['energies']))
    if np.round(np.log10(inFile['energies'][0]), 1) not in Energies:
        Energies.append(np.round(np.log10(inFile['energies'][0]), 1))
    xx = np.append(xx, np.array(inFile['xx']))
    yy = np.append(yy, np.array(inFile['yy']))
    zz = np.append(zz, np.array(inFile['zz']))
    flavors = np.append(flavors, np.array(inFile['flavors']))
    inelasticity = np.append(inelasticity, np.array(inFile['inelasticity']))
    interaction_type = np.append(interaction_type, np.array(inFile['interaction_type']))
    azimuths = np.append(azimuths, np.array(inFile['azimuths']))
    zeniths = np.append(zeniths, np.array(inFile['zeniths']))
    SNRs = np.append(SNRs, np.array(inFile['station_101']['SNRs']))
    max_amp_ray_solution = np.append(max_amp_ray_solution, np.array(inFile['station_101']['max_amp_ray_solution']), axis = 0)
    ray_tracing_solution_type = np.append(ray_tracing_solution_type, np.array(inFile['station_101']['ray_tracing_solution_type']), axis = 0)
    travel_times = np.append(travel_times, np.array(inFile['station_101']['travel_times']), axis = 0)
    travel_distances = np.append(travel_distances, np.array(inFile['station_101']['travel_distances']), axis = 0)
    receive_vectors = np.append(receive_vectors, np.array(inFile['station_101']['receive_vectors']), axis = 0)
    launch_vectors = np.append(launch_vectors, np.array(inFile['station_101']['launch_vectors']), axis = 0)
    polarization = np.append(polarization, np.array(inFile['station_101']['polarization']), axis = 0)
Energies.sort()
evtNum = len(event_ids)
interaction = np.zeros((evtNum, ), dtype = int)
# turn interaction type into 0/1
for i in range(evtNum):
    if interaction_type[i] == b'cc':
        interaction[i] = 1
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
showerEnergies = np.round(np.log10(np.where((np.abs(flavors) == 12) & (interaction == 1), 1., inelasticity) * energies), 2) # calculate shower energy
energies = np.round(np.log10(energies), 1)
zz = -zz #use positive depth as under surface
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

print("Data cleaning ...")
travel_times = travel_times.reshape(travel_times.shape[0] * channelPerStr * strNum, 2)
Filter = travel_times[:, 0] < travel_times[:, 1] # always put the ray with shorter travel time first
travel_times_dir = np.where(Filter, travel_times[:, 0], travel_times[:, 1])
travel_times_ref = np.where(Filter, travel_times[:, 1], travel_times[:, 0])
# turn into a 4*4 array with columns representing strings and rows representing antennas in topV botV topH botH
travel_times_dir = travel_times_dir.reshape(int(travel_times_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
travel_times_ref = travel_times_ref.reshape(int(travel_times_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
chan0TimeDir = travel_times_dir[:, 0, 0].reshape((travel_times_dir.shape[0], 1, 1)) + 100.
travel_times_dir -= chan0TimeDir # use channel 0 first ray as time origin
travel_times_ref -= chan0TimeDir
max_amp_ray_solution = np.absolute(max_amp_ray_solution) # use absolute value of peak
max_amp_ray_solution = max_amp_ray_solution.reshape(max_amp_ray_solution.shape[0] * channelPerStr * strNum, 2)
max_amp_ray_solution_dir = np.where(Filter, max_amp_ray_solution[:, 0], max_amp_ray_solution[:, 1])
max_amp_ray_solution_ref = np.where(Filter, max_amp_ray_solution[:, 1], max_amp_ray_solution[:, 0])
max_amp_ray_solution_dir = max_amp_ray_solution_dir.reshape(int(max_amp_ray_solution_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
max_amp_ray_solution_ref = max_amp_ray_solution_ref.reshape(int(max_amp_ray_solution_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
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
# apply true sign to amp to calculate ratio
max_amp_ray_solution_dir[:, 0:2, :] = max_amp_ray_solution_dir[:, 0:2, :] * np.sign(vAmp_dir[:, 0:2, :])
max_amp_ray_solution_dir[:, 2:4, :] = max_amp_ray_solution_dir[:, 2:4, :] * np.sign(hAmp_dir[:, 2:4, :])
max_amp_ray_solution_ref[:, 0:2, :] = max_amp_ray_solution_ref[:, 0:2, :] * np.sign(vAmp_ref[:, 0:2, :])
max_amp_ray_solution_ref[:, 2:4, :] = max_amp_ray_solution_ref[:, 2:4, :] * np.sign(hAmp_ref[:, 2:4, :])
# calculate amp ratio angle 0-2pi
ratio_amp_dir = np.arctan2(max_amp_ray_solution_dir[:, 2:4, :], max_amp_ray_solution_dir[:, 0:2, :])
ratio_amp_dir = np.where(ratio_amp_dir < 0, ratio_amp_dir + 2. * np.pi, ratio_amp_dir)
ratio_amp_ref = np.arctan2(max_amp_ray_solution_dir[:, 2:4, :], max_amp_ray_solution_dir[:, 0:2, :])
ratio_amp_ref = np.where(ratio_amp_ref < 0, ratio_amp_ref + 2. * np.pi, ratio_amp_ref)
# use the same angles for hpol and vpol in 4*4 array
ratio_amp_dir = np.repeat(ratio_amp_dir, 2, axis = 0).reshape(ratio_amp_dir.shape[0], channelPerStr, strNum)
ratio_amp_ref = np.repeat(ratio_amp_ref, 2, axis = 0).reshape(ratio_amp_ref.shape[0], channelPerStr, strNum)
# apply true sign to amp again to recover. so amps are now absolute values
max_amp_ray_solution_dir[:, 0:2, :] = max_amp_ray_solution_dir[:, 0:2, :] * np.sign(vAmp_dir[:, 0:2, :])
max_amp_ray_solution_dir[:, 2:4, :] = max_amp_ray_solution_dir[:, 2:4, :] * np.sign(hAmp_dir[:, 2:4, :])
max_amp_ray_solution_ref[:, 0:2, :] = max_amp_ray_solution_ref[:, 0:2, :] * np.sign(vAmp_ref[:, 0:2, :])
max_amp_ray_solution_ref[:, 2:4, :] = max_amp_ray_solution_ref[:, 2:4, :] * np.sign(hAmp_ref[:, 2:4, :])
max_amp_ray_solution_dir = normalize(max_amp_ray_solution_dir)
max_amp_ray_solution_ref = normalize(max_amp_ray_solution_ref)
# reshape them to apply Filter
cone_angles = cone_angles.reshape(cone_angles.shape[0] * channelPerStr * strNum, 2)
view_angles = view_angles.reshape(view_angles.shape[0] * channelPerStr * strNum, 2)
rec_theta = rec_theta.reshape(rec_theta.shape[0] * channelPerStr * strNum, 2)
rec_phi = rec_phi.reshape(rec_phi.shape[0] * channelPerStr * strNum, 2)
# put rays with shorter travel times first
cone_angles_dir = np.where(Filter, cone_angles[:, 0], cone_angles[:, 1])
cone_angles_ref = np.where(Filter, cone_angles[:, 1], cone_angles[:, 0])
view_angles_dir = np.where(Filter, view_angles[:, 0], view_angles[:, 1])
view_angles_ref = np.where(Filter, view_angles[:, 1], view_angles[:, 0])
rec_theta_dir = np.where(Filter, rec_theta[:, 0], rec_theta[:, 1])
rec_theta_ref = np.where(Filter, rec_theta[:, 1], rec_theta[:, 0])
rec_phi_dir = np.where(Filter, rec_phi[:, 0], rec_phi[:, 1])
rec_phi_ref = np.where(Filter, rec_phi[:, 1], rec_phi[:, 0])
# turn them back to event*antenna*str
cone_angles_dir = cone_angles_dir.reshape(int(cone_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
cone_angles_ref = cone_angles_ref.reshape(int(cone_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
view_angles_dir = view_angles_dir.reshape(int(view_angles_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
view_angles_ref = view_angles_ref.reshape(int(view_angles_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
rec_theta_dir = rec_theta_dir.reshape(int(rec_theta_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
rec_theta_ref = rec_theta_ref.reshape(int(rec_theta_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
rec_phi_dir = rec_phi_dir.reshape(int(rec_phi_dir.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
rec_phi_ref = rec_phi_ref.reshape(int(rec_phi_ref.shape[0] / channelPerStr / strNum), channelPerStr, strNum)
# use first channel as metric
viewAngle_dir = view_angles_dir[:, 0, 0].copy()
viewAngle_ref = view_angles_ref[:, 0, 0].copy()
coneAngle_dir = cone_angles_dir[:, 0, 0].copy()
coneAngle_ref = cone_angles_ref[:, 0, 0].copy()
# create a general x array
x = np.stack((travel_times_dir, travel_times_ref, max_amp_ray_solution_dir, max_amp_ray_solution_ref, ratio_amp_dir, ratio_amp_ref, view_angles_dir, view_angles_ref, cone_angles_dir, cone_angles_ref, rec_theta_dir, rec_theta_ref, rec_phi_dir, rec_phi_ref), axis = 3)
# create a general y array
y = np.vstack((rr, zz, dd, pp, tt, cos, sin, azimuths, zeniths, energies, cosAz, sinAz, showerEnergies, xx, yy, flavors, viewAngle_dir, viewAngle_ref, coneAngle_dir, coneAngle_ref, shower_axis[:, 0], shower_axis[:, 1], shower_axis[:, 2], launch_vectors_dir[:, 0], launch_vectors_dir[:, 1], launch_vectors_dir[:, 2]))
y = np.transpose(y)
maskY = np.isnan(y).any(axis = 1) # nan ray solutions filter
maskX = np.isnan(x).any(axis = 1).any(axis = 1).any(axis = 1) # nan ray solutions filter
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
mask = np.logical_or(np.logical_or(np.logical_or(maskX, maskY), maskCone), maskView)
x = x[~mask]
y = y[~mask]

print("Data spliting ...")
kfold = KFold(n_splits = numSplit, shuffle = True, random_state = 1) #split data into numSplit parts
kth = 0
for train, test in kfold.split(x, y):
    if kth != fold: # unless doing cross validation, always use the 0th fold
        kth += 1
        continue
    x_train = x[train]
    x_test = x[test]
    y_train = y[train]
    y_test = y[test]
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1, shuffle = True) # split test and validation evently so that 90% training, 5% validation, 5% test
    # normalize features
    xMean = np.nanmean(x_train, axis = 0)
    xStd = np.nanstd(x_train, axis = 0)
    xMin = np.nanmin(x_train, axis = 0)
    xMax = np.nanmax(x_train, axis = 0)
    x_train = (x_train - xMean) / xStd
    x_val = (x_val - xMean) / xStd
    x_test = (x_test - xMean) / xStd
    if Pred == "train":
        plotFeature(x_test, y_test)
    # select from the general x what features to use
    x_train = np.stack((x_train[:, :, :, 0], x_train[:, :, :, 1], x_train[:, :, :, 2], x_train[:, :, :, 3], x_train[:, :, :, 4], x_train[:, :, :, 5]), axis = -1)
    x_val = np.stack((x_val[:, :, :, 0], x_val[:, :, :, 1], x_val[:, :, :, 2], x_val[:, :, :, 3], x_val[:, :, :, 4], x_val[:, :, :, 5]), axis = -1)
    x_test = np.stack((x_test[:, :, :, 0], x_test[:, :, :, 1], x_test[:, :, :, 2], x_test[:, :, :, 3], x_test[:, :, :, 4], x_test[:, :, :, 5]), axis = -1)
    
    print("Setting up ...")
    inputs = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    rr_branch = separate(layers, "rr", inputs)
    tt_branch = separate(layers, "tt", inputs)
    cos_branch = separate(layers, "cos", inputs)
    sin_branch = separate(layers, "sin", inputs)
    ze_branch = separate(layers, "ze", inputs)
    cosAz_branch = separate(layers, "cosAz", inputs)
    sinAz_branch = separate(layers, "sinAz", inputs)
    sh_branch = separate(layers, "sh", inputs)
    model = Model(inputs = inputs, outputs = [rr_branch, tt_branch, cos_branch, sin_branch, ze_branch, cosAz_branch, sinAz_branch, sh_branch])
    model.compile(loss = {"rr_output":my_mspe, "tt_output":"mse", "cos_output":"mse", "sin_output":"mse", "ze_output":"mse", "cosAz_output":"mse", "sinAz_output":"mse", "sh_output":"mse"}, optimizer = "adam", loss_weights = {"rr_output":100., "tt_output":10000., "cos_output":10000., "sin_output":10000., "ze_output":100., "cosAz_output":100., "sinAz_output":100., "sh_output":1.})
    # save the best weight set
    checkpoint = ModelCheckpoint("./plots/nnRecon/allPairsWeights_{}layers{}nodes{}epochs{}batch{}fold.hdf5".format(layers, nodes, epochs, batch, fold), save_best_only = True, verbose = 1, monitor = 'val_loss', mode = 'min')
    model.summary()
    # plot architecture
    keras.utils.plot_model(model, "./plots/nnRecon/arch_{}layers{}nodes{}epochs{}batch{}fold.pdf".format(layers, nodes, epochs, batch, fold), show_shapes = True)

    if Pred == "train":
        print("Training ...")
        history = model.fit(x_train, [y_train[:, 0], y_train[:, 4], y_train[:, 5], y_train[:, 6], y_train[:, 8], y_train[:, 10], y_train[:, 11], y_train[:, 12]], epochs = epochs, batch_size = batch, verbose = 1, validation_data = (x_val, [y_val[:, 0], y_val[:, 4], y_val[:, 5], y_val[:, 6], y_val[:, 8], y_val[:, 10], y_val[:, 11], y_val[:, 12]]), callbacks = [checkpoint, ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, verbose = 1, patience = 3)])
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
        plotImportance(model, x_test, y_test)
        sys.exit(1) # finish train and exit

print("Making predictions ...")
model.load_weights("./plots/nnRecon/allPairsWeights_{}layers{}nodes{}epochs{}batch{}fold.hdf5".format(layers, nodes, epochs, batch, fold))
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
en_train = y_train[:, 9]
en_test = y_test[:, 9]
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

print("Making plots ...")
meanPerEnergies = []
sdPerEnergies = []
meanPerEnergiesNoout = []
sdPerEnergiesNoout = []
me = 0
sd = 0
meNoout = 0
sdNoout = 0
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
# consider the boundry case at 0/360 deg
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
# find outliers
if Pred == "rr":
    outliers = np.where(np.abs(diff / y_test.reshape((len(y_test), 1))) > 0.5, True, False)
elif Pred == "tt" or Pred == "pp" or Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    outliers = np.where(np.abs(diff) > 5., True, False)
elif Pred == "en" or Pred == "sh":
    outliers = np.where(np.abs(diff) > 1., True, False)

#mean error 1dhist per energy
plt.rc('font', size = 5)
for i in range(len(Energies)):
    if Pred == "rr":
        outSelect = outliers[en_test == Energies[i]]
        sdPerEnergies.append(np.std(diff[en_test == Energies[i]] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))))
        meanPerEnergies.append(np.mean(diff[en_test == Energies[i]] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))))
        sdPerEnergiesNoout.append(np.std(diff[en_test == Energies[i]][~outSelect] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))[~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff[en_test == Energies[i]][~outSelect] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1))[~outSelect]))
        plt.subplot(3, len(Energies) / 3 + 1, i + 1)
        plt.hist(diff[en_test == Energies[i]] / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1)), bins = 200, range = (-1.0, 1.01), density = True)
        plt.xlim((-1, 1))
        plt.ylim((0, 6))
        plt.xlabel("{}_relativeError".format(Pred))
    elif Pred == "tt" or Pred == "pp":
        outSelect = outliers[en_test == Energies[i]]
        sdPerEnergies.append(np.std(diff[en_test == Energies[i]]))
        meanPerEnergies.append(np.mean(diff[en_test == Energies[i]]))
        sdPerEnergiesNoout.append(np.std(diff[en_test == Energies[i]][~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff[en_test == Energies[i]][~outSelect]))
        plt.subplot(3, len(Energies) / 3 + 1, i + 1)
        plt.hist(diff[en_test == Energies[i]], bins = 200, range = (-5.0, 5.01), density = True)
        plt.xlim((-5, 5))
        plt.ylim((0, 2))
        plt.xlabel("{}_error[deg]".format(Pred))
    elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        outSelect = outliers[en_test == Energies[i]]
        sdPerEnergies.append(np.std(diff[en_test == Energies[i]]))
        meanPerEnergies.append(np.mean(diff[en_test == Energies[i]]))
        sdPerEnergiesNoout.append(np.std(diff[en_test == Energies[i]][~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff[en_test == Energies[i]][~outSelect]))
        plt.subplot(3, len(Energies) / 3 + 1, i + 1)
        plt.hist(diff[en_test == Energies[i]], bins = 200, range = (-15.0, 15.01), density = True)
        plt.xlim((-15, 15))
        plt.ylim((0, 0.4))
        plt.xlabel("{}_error[deg]".format(Pred))
    elif Pred == "sh":
        outSelect = outliers[en_test == Energies[i]]
        sdPerEnergies.append(np.std(diff[en_test == Energies[i]]))
        meanPerEnergies.append(np.mean(diff[en_test == Energies[i]]))
        sdPerEnergiesNoout.append(np.std(diff[en_test == Energies[i]][~outSelect]))
        meanPerEnergiesNoout.append(np.mean(diff[en_test == Energies[i]][~outSelect]))
        plt.subplot(3, len(Energies) / 3 + 1, i + 1)
        plt.hist(diff[en_test == Energies[i]], bins = 200, range = (-5.0, 5.01), density = True)
        plt.xlim((-5, 5))
        plt.ylim((0, 1))
        plt.xlabel("{}_error[log10(eV)]".format(Pred))
    plt.ylabel("{}_count".format(Pred))
    plt.grid(True)
    plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sd = {:.3f}\n{}_mean = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanErrorPerEnergy_1dHist_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#mean error 1dhist all
plt.rc('font', size = 10)
if Pred == "rr":
    me = np.mean(diff / y_test.reshape((len(y_test), 1)))
    sd = np.std(diff / y_test.reshape((len(y_test), 1)))
    meNoout = np.mean(diff[~outliers] / y_test.reshape((len(y_test), 1))[~outliers])
    sdNoout = np.std(diff[~outliers] / y_test.reshape((len(y_test), 1))[~outliers])
    plt.hist(diff / y_test.reshape((len(y_test), 1)), bins = 200, range = (-1.0, 1.01), density = True)
    plt.xlim((-1, 1))
    plt.ylim((0, 8))
    plt.xlabel("{}_relativeError".format(Pred))
elif Pred == "tt" or Pred == "pp":
    me = np.mean(diff)
    sd = np.std(diff)
    meNoout = np.mean(diff[~outliers])
    sdNoout = np.std(diff[~outliers])
    plt.hist(diff, bins = 200, range = (-5.0, 5.01), density = True)
    plt.xlim((-5, 5))
    plt.ylim((0, 2))
    plt.xlabel("{}_error[deg]".format(Pred))
elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    me = np.mean(diff)
    sd = np.std(diff)
    meNoout = np.mean(diff[~outliers])
    sdNoout = np.std(diff[~outliers])
    plt.hist(diff, bins = 200, range = (-15.0, 15.01), density = True)
    plt.xlim((-15, 15))
    plt.ylim((0, 0.4))
    plt.xlabel("{}_error[deg]".format(Pred))
elif Pred == "sh":
    me = np.mean(diff)
    sd = np.std(diff)
    meNoout = np.mean(diff[~outliers])
    sdNoout = np.std(diff[~outliers])
    plt.hist(diff, bins = 200, range = (-5.0, 5.01), density = True)
    plt.xlim((-5, 5))
    plt.ylim((0, 1))
    plt.xlabel("{}_error[log10(eV)]".format(Pred))
plt.ylabel("{}_count".format(Pred))
plt.grid(True)
plt.title("trained on {} samples\ntested on {} samples\n{}_sd = {:.3f}\n{}_mean = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(len(y_train), len(y_test), Pred, sdNoout, Pred, meNoout, layers, nodes, epochs, batch, fold))
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_1dHist_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
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
    plt.title("10^{:.3f} eV\n{}_{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], Pred, layers, nodes, epochs, batch, fold))
    plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("./plots/nnRecon/outliers_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#true vs pred
plt.rc('font', size = 5)
for i in range(len(Energies)):
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    plt.plot([0, int(max(y_test[en_test == Energies[i]]))], [0, int(max(y_test[en_test == Energies[i]]))], c = "r", linewidth = 0.5)
    plt.scatter(y_test[en_test == Energies[i]], y_test_pred[en_test == Energies[i]], s = 0.5)
    if Pred == "rr":
        plt.xlabel("{}_test[m]".format(Pred))
        plt.ylabel("{}_pred[m]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdRelativeError = {:.3f}\n{}_meanRelativeError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
    elif Pred == "en" or Pred == "sh":
        plt.xlabel("{}_test[log10(eV)]".format(Pred))
        plt.ylabel("{}_pred[log10(eV)]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
    elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylabel("{}_pred[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
plt.tight_layout()
plt.savefig("./plots/nnRecon/testVsPred_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

# diff vs true
plt.rc('font', size = 5)
for i in range(len(Energies)):
    diff = np.array(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1)) - y_test_pred[en_test == Energies[i]])
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    if Pred == "rr":
        plt.scatter(y_test[en_test == Energies[i]], diff / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1)), s = 0.5)
        plt.ylabel("{}_relative error".format(Pred))
        plt.xlabel("{}_test[m]".format(Pred))
        plt.ylim((-2.0, 2.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdRelativeError = {:.3f}\n{}_meanRelativeError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
    elif Pred == "tt" or Pred == "pp":
        plt.scatter(y_test[en_test == Energies[i]], diff, s = 0.5)
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylim((-5.0, 5.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
    elif Pred == "ze" or Pred == "az" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        plt.scatter(y_test[en_test == Energies[i]], diff, s = 0.5)
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylim((-15.0, 15.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
    elif Pred == "en" or Pred == "sh":
        plt.scatter(y_test[en_test == Energies[i]], diff, s = 0.5)
        plt.ylabel("{}_error[log10(eV)]".format(Pred))
        plt.xlabel("{}_test[log10(eV)]".format(Pred))
        plt.ylim((-2.0, 2.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}layers{}nodes{}epochs{}batch{}fold".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], layers, nodes, epochs, batch, fold))
    plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/testVsDiff_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#2dhist of mean error for training set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_rzHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#xy 2dhist of mean error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = (diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_xyHistLinear_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#xy 2dhist of mean error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_xyHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#direction 2dhist of mean error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_dirHistLinear_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#direction 2dhist of mean error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_dirHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of mean error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Dir[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_viewVsConeDirHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of rms error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 5., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Dir[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_viewVsConeDirHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of mean error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Ref[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_viewVsConeRefHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of rms error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 5., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 2., cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("Cone Angle Ref[deg]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_viewVsConeRefHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#2dhist of mean error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = (diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -5., vmax = 5., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = diff.reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -2., vmax = 2., cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_error[log10(eV)]".format(Pred))
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/meanError_rzHistLinear_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#xy 2dhist of rms error test
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_test.reshape((y_test.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_xyHistLinear_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#direction 2dhist of rms error test
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_test.reshape((y_test.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_dirHistLinear_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#xy 2dhist of rms error for train set
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_xyHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#direction 2dhist of rms error for train set
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("az[deg]")
plt.ylabel("ze[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_dirHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#2dhist of rms error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_rzHistLinear_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()

#2dhist of rms error linear for train set
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "pp" or Pred == "az":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "rr":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((len(y_train), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[deg]".format(Pred))
    plt.title("error = true - pred")
elif Pred == "en" or Pred == "sh":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 2, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsError[log10(eV)]".format(Pred))
    plt.title("error = true - pred")
plt.gca().set_aspect("equal")
plt.xlabel("rr[m]")
plt.ylabel("zz[m]")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/nnRecon/rmsError_rzHistLinear_forTrain{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
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
    plt.ylim(0., 1.)
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    plt.ylabel('{}_sdError[deg]'.format(Pred))
    plt.ylim(0., 5.)
elif Pred == "sh":
    plt.ylabel('{}_sdError[log10(eV)]'.format(Pred))
    plt.ylim(0., 5.)
plt.legend()
plt.subplot(2, 1, 2)
plt.grid(True)
plt.xlabel('Energies[log10(eV)]')
plt.plot(Energies, meanPerEnergies, marker = "x", label = "meanPerEnergies")
plt.plot(Energies, meanPerEnergiesNoout, marker = "o", label = "meanPerEnergiesNoout")
if Pred == "rr":
    plt.ylabel('{}_meanRelativeError'.format(Pred))
    plt.ylim(-0.2, 0.2)
elif Pred == "pp" or Pred == "tt" or Pred == "az" or Pred == "ze" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    plt.ylabel('{}_meanError[deg]'.format(Pred))
    plt.ylim(-2, 2)
elif Pred == "sh":
    plt.ylabel('{}_meanError[log10(eV)]'.format(Pred))
    plt.ylim(-2, 2)
plt.legend()
plt.suptitle("test_{}layers{}nodes{}epochs{}batch{}fold_{}".format(layers, nodes, epochs, batch, fold, Pred))
plt.tight_layout()
plt.savefig("./plots/nnRecon/test_{}layers{}nodes{}epochs{}batch{}fold_{}.pdf".format(layers, nodes, epochs, batch, fold, Pred))
plt.clf()
