print("Importing packages ...")
import numpy as np
import scipy.stats as stats
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
from scipy.stats import lognorm, norm, sem
from NuRadioMC.utilities import medium
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
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
from NuRadioMC.SignalProp import analyticraytracing as ray
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

def my_mspe(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.square((y_true - y_pred) / K.maximum(math_ops.abs(y_true), K.epsilon()))
    return K.mean(diff, axis = -1)

def normalize(mat):
    nRow = mat.shape[1]
    nCol = mat.shape[2]
    mat = mat.reshape((mat.shape[0], nRow * nCol))
    maximum = np.nanmax(mat, axis = 1)
    maximum = maximum.reshape((maximum.shape[0], 1))
    minimum = np.nanmin(mat, axis = 1)
    minimum = minimum.reshape((minimum.shape[0], 1))
    nume = np.maximum(maximum - minimum, 1e-10)
    mat = (mat - minimum) / nume
    return mat.reshape((mat.shape[0], nRow, nCol))

def share(layers, inputs):
    if layers == "2layers":
        x = Conv2D(32, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(inputs)
        x = Conv2D(16, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
    elif layers == "3layers":
        x = Conv2D(32, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(inputs)
        x = Conv2D(16, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Conv2D(8, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
    elif layers == "4layers":
        x = Conv2D(32, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(inputs)
        x = Conv2D(16, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Conv2D(8, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Conv2D(4, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
    else:
        sys.exit(1)
    x = Flatten()(x)
    #no shared convolutional layers, return inputs directly
    return inputs

def separate(layers, pred, shares):
    if layers == "2layers":
        x = Conv2D(32, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(shares)
        x = Conv2D(8, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation='relu')(x)
        x = Dense(1, name = "{}_output".format(pred))(x)
    elif layers == "3layers":
        x = Conv2D(32, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(shares)
        x = Conv2D(16, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Conv2D(8, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation='relu')(x)
        x = Dense(16, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = 'relu')(x)
        x = Dense(1, name = "{}_output".format(pred))(x)
    elif layers == "4layers":
        x = Conv2D(32, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(shares)
        x = Conv2D(16, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Conv2D(8, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Conv2D(4, kernel_size=(3, 3), padding = "same", kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = "relu")(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation='relu')(x)
        x = Dense(16, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = 'relu')(x)
        x = Dense(8, kernel_initializer = keras.initializers.he_uniform(seed = 1), activation = 'relu')(x)
        x = Dense(1, name = "{}_output".format(pred))(x)
    else:
        sys.exit(1)
    return x

def plotLearn(pred):
    if pred == "rr" or pred == "zz" or pred == "dd" or pred == "cos" or pred == "sin" or pred == "cosAz" or pred == "sinAz":
        plt.plot(np.sqrt(np.array(history.history["{}_output_loss".format(pred)])), label = "training", linewidth = 0.5)
        plt.plot(np.sqrt(np.array(history.history["val_{}_output_loss".format(pred)])), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 0.5))
        plt.yscale('log')
    elif pred == "en" or pred == "sh":
        plt.plot(np.sqrt(np.array(history.history["{}_output_loss".format(pred)])), label = "training", linewidth = 0.5)
        plt.plot(np.sqrt(np.array(history.history["val_{}_output_loss".format(pred)])), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 5))
        plt.yscale('log')
    elif pred == "tt" or pred == "rt" or pred == "ze":
        plt.plot(np.degrees(np.sqrt(np.array(history.history["{}_output_loss".format(pred)]))), label = "training", linewidth = 0.5)
        plt.plot(np.degrees(np.sqrt(np.array(history.history["val_{}_output_loss".format(pred)]))), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 45))
        plt.yscale('log')
    elif pred == "total":
        plt.plot(np.array(history.history["loss"]), label = "training", linewidth = 0.5)
        plt.plot(np.array(history.history["val_loss"]), label = "validation", linewidth = 0.5)
        plt.ylim((0.01, 45))
        plt.yscale('log')
    elif pred == "combine":
        plt.plot(np.array(history.history["loss"]), label = "loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["rr_output_loss"]), label = "rr_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["zz_output_loss"]), label = "zz_output_loss", linewidth = 1)
        plt.plot(10000. * np.array(history.history["cos_output_loss"]), label = "cos_output_loss", linewidth = 1)
        plt.plot(10000. * np.array(history.history["sin_output_loss"]), label = "sin_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["cosAz_output_loss"]), label = "cosAz_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["sinAz_output_loss"]), label = "sinAz_output_loss", linewidth = 1)
        plt.plot(np.array(history.history["sh_output_loss"]), label = "sh_output_loss", linewidth = 1)
        plt.plot(10000. * np.array(history.history["tt_output_loss"]), label = "tt_output_loss", linewidth = 1)
        plt.plot(100. * np.array(history.history["ze_output_loss"]), label = "ze_output_loss", linewidth = 1)
        plt.ylim((0.01, 1000))
        plt.yscale('log')
    plt.title("train on {} samples\nvalidate on {} samples\n{}".format(len(y_train), len(y_val), postFix))        
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("{}_loss".format(pred))
    plt.tight_layout()
    plt.savefig("./plots/covRecon/loss_{}_{}train0test0.pdf".format(postFix, pred))
    plt.clf()

Energies = np.array([16.5, 17.0, 17.5, 18, 18.5, 19.0, 19.5, 20.])
postFix = sys.argv[-1]
print("Modeling: {}".format(postFix))
Pred = sys.argv[-2]
print("Ploting: {}".format(Pred))
inFile = h5py.File(sys.argv[1], 'r')
print("Reading " + str(sys.argv[1]))
event_ids = np.array(inFile['event_ids']) + 1 * 10 ** 5
energies = np.array(inFile['energies'])
xx = np.array(inFile['xx'])
yy = np.array(inFile['yy'])
zz = np.array(inFile['zz'])
flavors = np.array(inFile['flavors'])
inelasticity = np.array(inFile['inelasticity'])
interaction_type = np.array(inFile['interaction_type'])
azimuths = np.array(inFile['azimuths'])
zeniths = np.array(inFile['zeniths'])
max_amp_ray_solution = np.array(inFile['station_101']['max_amp_ray_solution'])
ray_tracing_solution_type = np.array(inFile['station_101']['ray_tracing_solution_type'])
travel_times = np.array(inFile['station_101']['travel_times'])
travel_distances = np.array(inFile['station_101']['travel_distances'])
receive_vectors = np.array(inFile['station_101']['receive_vectors'])
launch_vectors = np.array(inFile['station_101']['launch_vectors'])
polarization = np.array(inFile['station_101']['polarization'])
antenna_positions = inFile['station_101'].attrs['antenna_positions']
Vrms = inFile.attrs['Vrms']
for i in range(2, len(sys.argv) - 2):
    inFile = h5py.File(sys.argv[i], 'r')
    print("Reading " + str(sys.argv[i]))
    event_ids = np.append(event_ids, np.array(inFile['event_ids']) + ((i - 2) / 8 + 1) * 10**5)
    energies = np.append(energies, np.array(inFile['energies']))
    xx = np.append(xx, np.array(inFile['xx']))
    yy = np.append(yy, np.array(inFile['yy']))
    zz = np.append(zz, np.array(inFile['zz']))
    flavors = np.append(flavors, np.array(inFile['flavors']))
    inelasticity = np.append(inelasticity, np.array(inFile['inelasticity']))
    interaction_type = np.append(interaction_type, np.array(inFile['interaction_type']))
    azimuths = np.append(azimuths, np.array(inFile['azimuths']))
    zeniths = np.append(zeniths, np.array(inFile['zeniths']))
    max_amp_ray_solution = np.append(max_amp_ray_solution, np.array(inFile['station_101']['max_amp_ray_solution']), axis = 0)
    ray_tracing_solution_type = np.append(ray_tracing_solution_type, np.array(inFile['station_101']['ray_tracing_solution_type']), axis = 0)
    travel_times = np.append(travel_times, np.array(inFile['station_101']['travel_times']), axis = 0)
    travel_distances = np.append(travel_distances, np.array(inFile['station_101']['travel_distances']), axis = 0)
    receive_vectors = np.append(receive_vectors, np.array(inFile['station_101']['receive_vectors']), axis = 0)
    launch_vectors = np.append(launch_vectors, np.array(inFile['station_101']['launch_vectors']), axis = 0)
    polarization = np.append(polarization, np.array(inFile['station_101']['polarization']), axis = 0)
    Vrms = inFile.attrs['Vrms']
strNum = 4
channelPerStr = 4
evtNum = len(event_ids)
shower_axis = -1.0 * hp.spherical_to_cartesian(zeniths, azimuths)
viewing_angles_dir = np.zeros((evtNum, strNum * channelPerStr))
viewing_angles_ref = np.zeros((evtNum, strNum * channelPerStr))
viewing_angles_dir[:, 0] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])
viewing_angles_ref[:, 0] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 1])])
for i in range(1, strNum * channelPerStr):
    viewing_angles_dir[:, i] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, i, 0])])
    viewing_angles_ref[:, i] = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, i, 1])])
viewing_angles_dir = viewing_angles_dir.reshape(viewing_angles_dir.shape[0], channelPerStr, strNum)
viewing_angles_ref = viewing_angles_ref.reshape(viewing_angles_ref.shape[0], channelPerStr, strNum)
cone_angles_dir = np.zeros((evtNum, strNum * channelPerStr))
cone_angles_ref = np.zeros((evtNum, strNum * channelPerStr))
zHat = np.array([0., 0., 1.])
launch_vectors_dir = launch_vectors[:, 0, 0, :]
launch_vectors_ref = launch_vectors[:, 0, 1, :]
yOnCone = np.cross(zHat, shower_axis)
yOnCone = yOnCone / np.linalg.norm(yOnCone, axis = 1).reshape((yOnCone.shape[0], 1))
zOnCone = np.cross(shower_axis, yOnCone)
zOnCone = zOnCone / np.linalg.norm(zOnCone, axis = 1).reshape((zOnCone.shape[0], 1))
launchOnCone_dir = launch_vectors_dir - shower_axis * np.sum(launch_vectors_dir * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
launchOnCone_dir = launchOnCone_dir / np.linalg.norm(launchOnCone_dir, axis = 1).reshape((launchOnCone_dir.shape[0], 1))
cone_angles_dir[:, 0] = np.arccos(np.sum(launchOnCone_dir * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_dir * yOnCone, axis = 1))
launchOnCone_ref = launch_vectors_ref - shower_axis * np.sum(launch_vectors_ref * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
launchOnCone_ref = launchOnCone_ref / np.linalg.norm(launchOnCone_ref, axis = 1).reshape((launchOnCone_ref.shape[0], 1))
cone_angles_ref[:, 0] = np.arccos(np.sum(launchOnCone_ref * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_ref * yOnCone, axis = 1))
for i in range(1, strNum * channelPerStr):
    launch_vectors_dir = launch_vectors[:, i, 0, :]
    launch_vectors_ref = launch_vectors[:, i, 1, :]
    yOnCone = np.cross(zHat, shower_axis)
    yOnCone = yOnCone / np.linalg.norm(yOnCone, axis = 1).reshape((yOnCone.shape[0], 1))
    zOnCone = np.cross(shower_axis, yOnCone)
    zOnCone = zOnCone / np.linalg.norm(zOnCone, axis = 1).reshape((zOnCone.shape[0], 1))
    launchOnCone_dir = launch_vectors_dir - shower_axis * np.sum(launch_vectors_dir * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
    launchOnCone_dir = launchOnCone_dir / np.linalg.norm(launchOnCone_dir, axis = 1).reshape((launchOnCone_dir.shape[0], 1))
    cone_angles_dir[:, i] = np.arccos(np.sum(launchOnCone_dir * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_dir * yOnCone, axis = 1))
    launchOnCone_ref = launch_vectors_ref - shower_axis * np.sum(launch_vectors_ref * shower_axis, axis = 1).reshape((shower_axis.shape[0], 1))
    launchOnCone_ref = launchOnCone_ref / np.linalg.norm(launchOnCone_ref, axis = 1).reshape((launchOnCone_ref.shape[0], 1))
    cone_angles_ref[:, i] = np.arccos(np.sum(launchOnCone_ref * zOnCone, axis = 1)) * np.sign(np.sum(launchOnCone_ref * yOnCone, axis = 1))
cone_angles_dir = cone_angles_dir.reshape(cone_angles_dir.shape[0], channelPerStr, strNum)
cone_angles_ref = cone_angles_ref.reshape(cone_angles_ref.shape[0], channelPerStr, strNum)
ice = medium.southpole_2015()
n_index = np.array([ice.get_index_of_refraction(x) for x in np.array([xx, yy, zz]).T])
cherenkov = np.arccos(1. / n_index)
hAmp = np.sign(np.multiply(polarization[:, :, :, 0], polarization[:, :, :, 1])) * np.sqrt(np.add(np.square(polarization[:, :, :, 0]), np.square(polarization[:, :, :, 1])))
vAmp = polarization[:, :, :, 2]
ratio = np.arctan2(hAmp, vAmp)
showerEnergies = np.round(np.log10(np.where((np.abs(flavors) == 12) & (interaction_type == "cc"), 1., inelasticity) * energies), 2)
energies = np.round(np.log10(energies), 1)
zz = -zz #use positive depth as under surface
rr = np.sqrt(np.square(xx) + np.square(yy))
tt = -1. * np.arcsin((zz - 200.) / np.sqrt(np.square(xx) + np.square(yy) + np.square(zz - 200.)))
pp = np.arctan2(yy, xx)
pp = np.where(pp < 0, pp + 2 * np.pi, pp)
dd = np.sqrt(np.square(rr) + np.square(zz - 200.))
rt, rp = hp.cartesian_to_spherical(receive_vectors[:, :, 0, 0].flatten(), receive_vectors[:, :, 0, 1].flatten(), receive_vectors[:, :, 0, 2].flatten())#receive_vectors[evt, chan, dir/ref, xyz]
rt = -1. * np.mean(rt.reshape(receive_vectors.shape[0], strNum * channelPerStr), axis = 1) + np.pi / 2.
cos = xx / rr
sin = yy / rr
cosAz = np.cos(azimuths)
sinAz = np.sin(azimuths)

#nn data cleaning
print("Data cleaning ...")
travel_times = travel_times.reshape(travel_times.shape[0] * strNum * channelPerStr, 2)
Filter = travel_times[:, 0] < travel_times[:, 1]
travel_times_dir = np.where(Filter, travel_times[:, 0], travel_times[:, 1])
travel_times_ref = np.where(Filter, travel_times[:, 1], travel_times[:, 0])
travel_times_dir = travel_times_dir.reshape(int(travel_times_dir.shape[0] / strNum / channelPerStr), channelPerStr, strNum)
travel_times_ref = travel_times_ref.reshape(int(travel_times_ref.shape[0] / strNum / channelPerStr), channelPerStr, strNum)
chan0TimeDir = travel_times_dir[:, 0, 0].reshape((travel_times_dir.shape[0], 1, 1)) + 100.
travel_times_dir -= chan0TimeDir
travel_times_dir = normalize(travel_times_dir)
travel_times_dir_mask = (~np.isnan(travel_times_dir)).astype(float)
travel_times_ref -= chan0TimeDir
travel_times_ref = normalize(travel_times_ref)
travel_times_ref_mask = (~np.isnan(travel_times_ref)).astype(float)
max_amp_ray_solution = max_amp_ray_solution.reshape(max_amp_ray_solution.shape[0] * strNum * channelPerStr, 2)
max_amp_ray_solution_dir = np.where(Filter, max_amp_ray_solution[:, 0], max_amp_ray_solution[:, 1])
max_amp_ray_solution_ref = np.where(Filter, max_amp_ray_solution[:, 1], max_amp_ray_solution[:, 0])
max_amp_ray_solution_dir = max_amp_ray_solution_dir.reshape(int(max_amp_ray_solution_dir.shape[0] / strNum / channelPerStr), channelPerStr, strNum)
max_amp_ray_solution_ref = max_amp_ray_solution_ref.reshape(int(max_amp_ray_solution_ref.shape[0] / strNum / channelPerStr), channelPerStr, strNum)
max_amp_ray_solution_dir = normalize(max_amp_ray_solution_dir)
max_amp_ray_solution_dir_mask = (~np.isnan(max_amp_ray_solution_dir)).astype(float)
max_amp_ray_solution_ref = normalize(max_amp_ray_solution_ref)
max_amp_ray_solution_ref_mask = (~np.isnan(max_amp_ray_solution_ref)).astype(float)
ratio = ratio.reshape(ratio.shape[0] * strNum * channelPerStr, 2)
ratio_dir = np.where(Filter, ratio[:, 0], ratio[:, 1])
ratio_ref = np.where(Filter, ratio[:, 1], ratio[:, 0])
ratio_dir = ratio_dir.reshape(int(ratio_dir.shape[0] / strNum / channelPerStr), channelPerStr, strNum)
ratio_ref = ratio_ref.reshape(int(ratio_ref.shape[0] / strNum / channelPerStr), channelPerStr, strNum)
Filter = Filter.reshape(int(Filter.shape[0] / strNum / channelPerStr), strNum * channelPerStr)
viewAngle_dir, viewAngle_ref = np.where(Filter[:, 0], viewing_angles_dir[:, 0, 0], viewing_angles_ref[:, 0, 0]).copy(), np.where(Filter[:, 0], viewing_angles_ref[:, 0, 0], viewing_angles_dir[:, 0, 0]).copy()
coneAngle_dir, coneAngle_ref = np.where(Filter[:, 0], cone_angles_dir[:, 0, 0], cone_angles_ref[:, 0, 0]).copy(), np.where(Filter[:, 0], cone_angles_ref[:, 0, 0], cone_angles_dir[:, 0, 0]).copy()
#inputs
x = np.stack((travel_times_dir, travel_times_ref, max_amp_ray_solution_dir, max_amp_ray_solution_ref), axis = 3)
y = np.vstack((rr, zz, dd, pp, tt, cos, sin, azimuths, zeniths, energies, cosAz, sinAz, showerEnergies, xx, yy, flavors, viewAngle_dir, viewAngle_ref, coneAngle_dir, coneAngle_ref))
y = np.transpose(y)
maskY = np.isnan(y).any(axis = 1)
maskX = np.isnan(x).any(axis = 1).any(axis = 1).any(axis = 1)
ray_tracing_solution_type = ray_tracing_solution_type.reshape((ray_tracing_solution_type.shape[0], ray_tracing_solution_type.shape[1] * 2))
maskSolutionType = np.where(ray_tracing_solution_type == 3, True, False).any(axis = 1)
mask = np.logical_or(maskX, maskY)
x = x[~mask]
y = y[~mask]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1, shuffle = True)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1, shuffle = True)
xMean = np.nanmean(x_train, axis = 0)
xStd = np.nanstd(x_train, axis = 0)
xMin = np.nanmin(x_train, axis = 0)
xMax = np.nanmax(x_train, axis = 0)
x_train = (x_train - xMean) / xStd
x_val = (x_val - xMean) / xStd
x_test = (x_test - xMean) / xStd

#nn setup
print("Setting up ...")
inputs = Input(shape = (channelPerStr, strNum, 4))
shares = share(postFix, inputs)
rr_branch = separate(postFix, "rr", shares)
zz_branch = separate(postFix, "zz", shares)
tt_branch = separate(postFix, "tt", shares)
cos_branch = separate(postFix, "cos", shares)
sin_branch = separate(postFix, "sin", shares)
ze_branch = separate(postFix, "ze", shares)
cosAz_branch = separate(postFix, "cosAz", shares)
sinAz_branch = separate(postFix, "sinAz", shares)
sh_branch = separate(postFix, "sh", shares)
model = Model(inputs = inputs, outputs = [rr_branch, zz_branch, tt_branch, cos_branch, sin_branch, ze_branch, cosAz_branch, sinAz_branch, sh_branch])
model.compile(loss = {"rr_output":my_mspe, "zz_output":my_mspe, "tt_output":"mse", "cos_output":"mse", "sin_output":"mse", "ze_output":"mse", "cosAz_output":"mse", "sinAz_output":"mse", "sh_output":"mse"}, optimizer = "adam", loss_weights = {"rr_output":100., "zz_output":100., "tt_output":10000., "cos_output":10000., "sin_output":10000., "ze_output":100., "cosAz_output":100., "sinAz_output":100., "sh_output":1.})
checkpoint = ModelCheckpoint("./plots/covRecon/allPairsWeights_" + str(postFix) + "train0test0.hdf5", save_best_only = True, verbose = 1, monitor = 'val_loss', mode = 'min')
model.summary()
keras.utils.plot_model(model, "./plots/covRecon/arch_{}train0test0.pdf".format(postFix), show_shapes = True)

#nn training
if Pred == "train":
    print("Training ...")
    history = model.fit(x_train, [y_train[:, 0], y_train[:, 1], y_train[:, 4], y_train[:, 5], y_train[:, 6], y_train[:, 8], y_train[:, 10], y_train[:, 11], y_train[:, 12]], epochs = 50, batch_size = 128, verbose = 1, validation_data = (x_val, [y_val[:, 0], y_val[:, 1], y_val[:, 4], y_val[:, 5], y_val[:, 6], y_val[:, 8], y_val[:, 10], y_val[:, 11], y_val[:, 12]]), callbacks = [checkpoint, ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, verbose = 1, patience = 3)])
    plotLearn("combine")
    plotLearn("total")
    plotLearn("rr")
    plotLearn("zz")
    plotLearn("tt")
    plotLearn("cos")
    plotLearn("sin")
    plotLearn("ze")
    plotLearn("cosAz")
    plotLearn("sinAz")
    plotLearn("sh")
    sys.exit(1)

#nn analysis
print("Making plots ...")
model.load_weights("./plots/covRecon/allPairsWeights_{}train0test0.hdf5".format(postFix))
fl_train = y_train[:, 15].reshape((len(y_train), 1))
fl_test = y_test[:, 15].reshape((len(y_test), 1))
rr_train = y_train[:, 0].reshape((len(y_train), 1))
rr_test = y_test[:, 0].reshape((len(y_test), 1))
zz_train = y_train[:, 1].reshape((len(y_train), 1))
zz_test = y_test[:, 1].reshape((len(y_test), 1))
en_train = y_train[:, 9]
en_test = y_test[:, 9]
az_train = np.degrees(y_train[:, 7].reshape((len(y_train), 1)))
az_test = np.degrees(y_test[:, 7].reshape((len(y_test), 1)))
ze_train = np.degrees(y_train[:, 8].reshape((len(y_train), 1)))
ze_test = np.degrees(y_test[:, 8].reshape((len(y_test), 1)))
xx_train = y_train[:, 13].reshape((len(y_train), 1))
xx_test = y_test[:, 13].reshape((len(y_test), 1))
yy_train = y_train[:, 14].reshape((len(y_train), 1))
yy_test = y_test[:, 14].reshape((len(y_test), 1))
viewAngle_dir_train = np.degrees(y_train[:, 16].reshape((len(y_train), 1)))
viewAngle_ref_train = np.degrees(y_train[:, 17].reshape((len(y_train), 1)))
coneAngle_dir_train = np.degrees(y_train[:, 18].reshape((len(y_train), 1)))
coneAngle_ref_train = np.degrees(y_train[:, 19].reshape((len(y_train), 1)))

if Pred == "cos" or Pred == "sin":
    y_test_pred = np.arctan2(np.array(model.predict(x_test, batch_size = 128))[4], np.array(model.predict(x_test, batch_size = 128))[3])
    y_test_pred = np.where(y_test_pred < 0, y_test_pred + 2 * np.pi, y_test_pred)
    y_train_pred = np.arctan2(np.array(model.predict(x_train, batch_size = 128))[4], np.array(model.predict(x_train, batch_size = 128))[3])
    y_train_pred = np.where(y_train_pred < 0, y_train_pred + 2 * np.pi, y_train_pred)
    y_test = y_test[:, 3]
    y_train = y_train[:, 3]
elif Pred == "cosAz" or Pred == "sinAz":
    y_test_pred = np.arctan2(np.array(model.predict(x_test, batch_size = 128))[7], np.array(model.predict(x_test, batch_size = 128))[6])
    y_test_pred = np.where(y_test_pred < 0, y_test_pred + 2 * np.pi, y_test_pred)
    y_train_pred = np.arctan2(np.array(model.predict(x_train, batch_size = 128))[7], np.array(model.predict(x_train, batch_size = 128))[6])
    y_train_pred = np.where(y_train_pred < 0, y_train_pred + 2 * np.pi, y_train_pred)
    y_test = y_test[:, 7]
    y_train = y_train[:, 7]
elif Pred == "rr":
    y_test_pred = np.array(model.predict(x_test, batch_size = 128))[0]
    y_train_pred = np.array(model.predict(x_train, batch_size = 128))[0]
    y_test = y_test[:, 0]
    y_train = y_train[:, 0]
elif Pred == "zz":
    y_test_pred = np.array(model.predict(x_test, batch_size = 128))[1]
    y_train_pred = np.array(model.predict(x_train, batch_size = 128))[1]
    y_test = y_test[:, 1]
    y_train = y_train[:, 1]
elif Pred == "tt":
    y_test_pred = np.array(model.predict(x_test, batch_size = 128))[2]
    y_train_pred = np.array(model.predict(x_train, batch_size = 128))[2]
    y_test = y_test[:, 4]
    y_train = y_train[:, 4]
elif Pred == "ze":
    y_test_pred = np.array(model.predict(x_test, batch_size = 128))[5]
    y_train_pred = np.array(model.predict(x_train, batch_size = 128))[5]
    y_test = y_test[:, 8]
    y_train = y_train[:, 8]
elif Pred == "sh":
    y_test_pred = np.array(model.predict(x_test, batch_size = 128))[8]
    y_train_pred = np.array(model.predict(x_train, batch_size = 128))[8]
    y_test = y_test[:, 12]
    y_train = y_train[:, 12]
else:
    sys.exit(1)
if Pred == "tt" or Pred == "cos" or Pred == "sin" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz":
    y_test_pred = np.degrees(y_test_pred)
    y_train_pred = np.degrees(y_train_pred)
    y_train = np.degrees(y_train)
    y_test = np.degrees(y_test)
meanPerEnergies = []
sdPerEnergies = []
meanPerEnergiesNoout = []
sdPerEnergiesNoout = []
me = 0
sd = 0
meNoout = 0
sdNoout = 0
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "dd" or Pred == "rr":
    outliers = np.where(np.abs(diff / y_test.reshape((len(y_test), 1))) > 0.5, True, False)
elif Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    outliers = np.where(np.abs(diff) > 5., True, False)
elif Pred == "en" or Pred == "sh":
    outliers = np.where(np.abs(diff) > 1., True, False)

#mean error 1dhist per energy
plt.rc('font', size = 5)
for i in range(len(Energies)):
    if Pred == "zz" or Pred == "dd" or Pred == "rr":
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
    elif Pred == "tt" or Pred == "cos":
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
    elif Pred == "ze" or Pred == "cosAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
    plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sd = {:.3f}\n{}_mean = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
plt.tight_layout()
plt.savefig("./plots/covRecon/meanErrorPerEnergy_1dHist_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#mean error 1dhist
plt.rc('font', size = 10)
if Pred == "zz" or Pred == "dd" or Pred == "rr":
    me = np.mean(diff / y_test.reshape((len(y_test), 1)))
    sd = np.std(diff / y_test.reshape((len(y_test), 1)))
    meNoout = np.mean(diff[~outliers] / y_test.reshape((len(y_test), 1))[~outliers])
    sdNoout = np.std(diff[~outliers] / y_test.reshape((len(y_test), 1))[~outliers])
    plt.hist(diff / y_test.reshape((len(y_test), 1)), bins = 200, range = (-1.0, 1.01), density = True)
    plt.xlim((-1, 1))
    plt.ylim((0, 8))
    plt.xlabel("{}_relativeError".format(Pred))
elif Pred == "tt" or Pred == "cos":
    me = np.mean(diff)
    sd = np.std(diff)
    meNoout = np.mean(diff[~outliers])
    sdNoout = np.std(diff[~outliers])
    plt.hist(diff, bins = 200, range = (-5.0, 5.01), density = True)
    plt.xlim((-5, 5))
    plt.ylim((0, 2))
    plt.xlabel("{}_error[deg]".format(Pred))
elif Pred == "ze" or Pred == "cosAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.title("trained on {} samples\ntested on {} samples\n{}_sd = {:.3f}\n{}_mean = {:.3f}\n{}".format(len(y_train), len(y_test), Pred, sdNoout, Pred, meNoout, postFix))
plt.tight_layout()
plt.savefig("./plots/covRecon/meanError_1dHist_{}_{}train0test0.pdf".format(postFix, Pred))
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
    plt.title("10^{:.3f} eV\n{}_{}".format(Energies[i], Pred, postFix))
    plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("./plots/covRecon/outliers_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#true vs pred
plt.rc('font', size = 5)
for i in range(len(Energies)):
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    plt.plot([0, int(max(y_test[en_test == Energies[i]]))], [0, int(max(y_test[en_test == Energies[i]]))], c = "r", linewidth = 0.5)
    plt.scatter(y_test[en_test == Energies[i]], y_test_pred[en_test == Energies[i]], s = 0.5)
    if Pred == "zz" or Pred == "rr" or Pred == "dd":
        plt.xlabel("{}_test[m]".format(Pred))
        plt.ylabel("{}_pred[m]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdRelativeError = {:.3f}\n{}_meanRelativeError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
    elif Pred == "en" or Pred == "sh":
        plt.xlabel("{}_test[log10(eV)]".format(Pred))
        plt.ylabel("{}_pred[log10(eV)]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
    elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylabel("{}_pred[deg]".format(Pred))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
plt.tight_layout()
plt.savefig("./plots/covRecon/testVsPred_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

# diff vs true
plt.rc('font', size = 5)
for i in range(len(Energies)):
    diff = np.array(y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1)) - y_test_pred[en_test == Energies[i]])
    plt.subplot(3, len(Energies) / 3 + 1, i + 1)
    if Pred == "zz" or Pred == "rr" or Pred == "dd":
        plt.scatter(y_test[en_test == Energies[i]], diff / y_test[en_test == Energies[i]].reshape((len(y_test[en_test == Energies[i]]), 1)), s = 0.5)
        plt.ylabel("{}_relative error".format(Pred))
        plt.xlabel("{}_test[m]".format(Pred))
        plt.ylim((-2.0, 2.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdRelativeError = {:.3f}\n{}_meanRelativeError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
    elif Pred == "tt" or Pred == "cos":
        plt.scatter(y_test[en_test == Energies[i]], diff, s = 0.5)
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylim((-5.0, 5.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
    elif Pred == "ze" or Pred == "cosAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
        plt.scatter(y_test[en_test == Energies[i]], diff, s = 0.5)
        plt.ylabel("{}_error[deg]".format(Pred))
        plt.xlabel("{}_test[deg]".format(Pred))
        plt.ylim((-15.0, 15.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
    elif Pred == "en" or Pred == "sh":
        plt.scatter(y_test[en_test == Energies[i]], diff, s = 0.5)
        plt.ylabel("{}_error[log10(eV)]".format(Pred))
        plt.xlabel("{}_test[log10(eV)]".format(Pred))
        plt.ylim((-2.0, 2.0))
        plt.title("10^{:.3f} eV\ntrained on {} samples\ntested on {} samples\n{}_sdError = {:.3f}\n{}_meanError = {:.3f}\n{}".format(Energies[i], len(y_train[en_train == Energies[i]]), len(y_test[en_test == Energies[i]]), Pred, sdPerEnergiesNoout[i], Pred, meanPerEnergiesNoout[i], postFix))
    plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/covRecon/testVsDiff_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#2dhist of mean error for validation set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/meanError_rzHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#xy 2dhist of mean error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = (diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/meanError_xyHistLinear_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#xy 2dhist of mean error for validation set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/meanError_xyHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#direction 2dhist of mean error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.xlabel("azimuth[deg]")
plt.ylabel("zenith[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/covRecon/meanError_dirHistLinear_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#direction 2dhist of mean error for validation set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.xlabel("azimuth[deg]")
plt.ylabel("zenith[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/covRecon/meanError_dirHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of mean error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/meanError_viewVsConeDirHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#viewing angle dir Vs cone angle 2dhist of rms error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_dir_train.reshape((len(coneAngle_dir_train),)), viewAngle_dir_train.reshape((len(viewAngle_dir_train),)), bins=[np.arange(-180, 181, 5), np.arange(0, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/rmsError_viewVsConeDirHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of mean error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = (diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/meanError_viewVsConeRefHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#viewing angle ref Vs cone angle 2dhist of rms error for train set linear
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(coneAngle_ref_train.reshape((len(coneAngle_ref_train),)), viewAngle_ref_train.reshape((len(viewAngle_ref_train),)), bins=[np.arange(-180, 181, 5), np.arange(10, 111, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-180, 180, 0, 110), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.ylabel("viewing angle[deg]")
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/rmsError_viewVsConeRefHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#2dhist of mean error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = (diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow((nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = -0.5, vmax = 0.5, cmap = cm.RdBu)
    cb = plt.colorbar()
    cb.set_label("{}_meanRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/meanError_rzHistLinear_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#xy 2dhist of rms error test
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_test.reshape((y_test.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_test.reshape((len(xx_test),)), yy_test.reshape((len(yy_test),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/rmsError_xyHistLinear_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#direction 2dhist of rms error test
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_test.reshape((y_test.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_test.reshape((len(az_test),)), ze_test.reshape((len(ze_test),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.xlabel("azimuth[deg]")
plt.ylabel("zenith[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/covRecon/rmsError_dirHistLinear_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#xy 2dhist of rms error train
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(xx_train.reshape((len(xx_train),)), yy_train.reshape((len(yy_train),)), bins=[np.arange(-8001, 8001, 200), np.arange(-8001, 8001, 200)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (-8000, 8000, -8000, 8000), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/rmsError_xyHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#direction 2dhist of rms error train
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((y_train.shape[0], 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(az_train.reshape((len(az_train),)), ze_train.reshape((len(ze_train),)), bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 360, 0, 180), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.xlabel("azimuth[deg]")
plt.ylabel("zenith[deg]")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/covRecon/rmsError_dirHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#2dhist of rms error linear
plt.rc('font', size = 10)
diff = np.array(y_test.reshape((len(y_test), 1)) - y_test_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_test.reshape((len(y_test), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_test.reshape((len(rr_test),)), zz_test.reshape((len(zz_test),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/rmsError_rzHistLinear_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#2dhist of rms error linear train
plt.rc('font', size = 10)
diff = np.array(y_train.reshape((len(y_train), 1)) - y_train_pred)
if Pred == "cos" or Pred == "cosAz":
    diff = np.where(diff > 180., diff - 360., diff)
    diff = np.where(diff < -180., diff + 360., diff)
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    nume = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'), weights = np.square(diff / y_train.reshape((len(y_train), 1))).reshape((diff.shape[0],)))
    deno = plt.hist2d(rr_train.reshape((len(rr_train),)), zz_train.reshape((len(zz_train),)) * -1., bins=[np.arange(0, 8001, 100), np.arange(-3000, 1, 100)], cmap = plt.get_cmap('Blues'))
    plt.clf()
    plt.imshow(np.sqrt(nume[0] / deno[0]).transpose()[::-1], extent = (0, 8000, -3000, 0), vmin = 0, vmax = 0.5, cmap = cm.Blues)
    cb = plt.colorbar()
    cb.set_label("{}_rmsRelativeError".format(Pred))
    plt.title("relative error = (true - pred) / true")
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
plt.savefig("./plots/covRecon/rmsError_rzHistLinear_forTrain{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()

#error summary
plt.rc('font', size = 10)
plt.subplot(2, 1, 1)
plt.grid(True)
plt.xlabel('Energies[log10(eV)]')
plt.plot(Energies, sdPerEnergies, marker = "x", label = "sdPerEnergies")
plt.plot(Energies, sdPerEnergiesNoout, marker = "o", label = "sdPerEnergiesNoout")
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    plt.ylabel('{}_sdRelativeError'.format(Pred))
    plt.ylim(0., 1.)
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
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
if Pred == "zz" or Pred == "rr" or Pred == "dd":
    plt.ylabel('{}_meanRelativeError'.format(Pred))
    plt.ylim(-0.2, 0.2)
elif Pred == "pp" or Pred == "tt" or Pred == "rt" or Pred == "cos" or Pred == "sin" or Pred == "az" or Pred == "ze" or Pred == "cosAz" or Pred == "sinAz" or Pred == "viewDir" or Pred == "viewRef" or Pred == "coneDir" or Pred == "coneRef":
    plt.ylabel('{}_meanError[deg]'.format(Pred))
    plt.ylim(-2, 2)
elif Pred == "sh":
    plt.ylabel('{}_meanError[log10(eV)]'.format(Pred))
    plt.ylim(-2, 2)
plt.legend()
plt.suptitle("test_{}_{}".format(postFix, Pred))
plt.tight_layout()
plt.savefig("./plots/covRecon/test_{}_{}train0test0.pdf".format(postFix, Pred))
plt.clf()
