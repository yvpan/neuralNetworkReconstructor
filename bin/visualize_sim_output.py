import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
from matplotlib import cm
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import h5py
import argparse
import os

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str, nargs = '+', help='path to NuRadioMC hdf5 simulation output')
args = parser.parse_args()
filename = os.path.splitext(os.path.basename(args.inputfilename[0]))[0]
dirname = os.path.dirname(args.inputfilename[0])
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)
fin = h5py.File(args.inputfilename[0], 'r')
print("Reading " + str(args.inputfilename[0]))
energies = np.array(fin['energies'])
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])
zeniths = np.array(fin['zeniths'])
azimuths = np.array(fin['azimuths'])
max_amp_ray_solution = np.array(fin['station_101']['max_amp_ray_solution'])
travel_times = np.array(fin['station_101']['travel_times'])
for i in range(len(args.inputfilename) - 1):
    fin = h5py.File(args.inputfilename[i + 1], 'r')
    print("Reading " + str(args.inputfilename[i + 1]))
    energies = np.append(energies, np.array(fin['energies']))
    xx = np.append(xx, np.array(fin['xx']))
    yy = np.append(yy, np.array(fin['yy']))
    zz = np.append(zz, np.array(fin['zz']))
    zeniths = np.append(zeniths, np.array(fin['zeniths']))
    azimuths = np.append(azimuths, np.array(fin['azimuths']))
    max_amp_ray_solution = np.append(max_amp_ray_solution, np.array(fin['station_101']['max_amp_ray_solution']), axis = 0)
    travel_times = np.append(travel_times, np.array(fin['station_101']['travel_times']), axis = 0)
max_amp_ray_solution = np.absolute(max_amp_ray_solution)
pp = np.arctan2(yy, xx)
pp = np.where(pp < 0, pp + 2. * np.pi, pp)
tt = -1. * np.arcsin((-1. * zz - 200.) / np.sqrt(np.square(xx) + np.square(yy) + np.square(-1. * zz - 200.)))
chan0Time = travel_times[:, 0, 0].reshape(travel_times.shape[0], 1, 1)
travel_times -= chan0Time

fig, ax = plt.subplots(1, 1)
rr = (xx ** 2 + yy ** 2) ** 0.5
plt.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 8001, 100), np.arange(-3500, 1, 100)], cmap=plt.get_cmap('Blues'))
cb = plt.colorbar()
cb.set_label("number of events")
ax.set_aspect('equal')
plt.xlabel("rr [m]")
plt.ylabel("zz [m]")
plt.grid(True)
fig.tight_layout()
plt.title('vertex distribution')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'simOutputVertexRZ.pdf'))
plt.clf()    

fig, ax = plt.subplots(1, 1)
plt.hist2d(xx / units.m, yy / units.m, bins=[np.arange(-8000, 8001, 200), np.arange(-8000, 8001, 200)], cmap=plt.get_cmap('Blues'))
cb = plt.colorbar()
cb.set_label("number of events")
ax.set_aspect('equal')
plt.xlabel("xx [m]")
plt.ylabel("yy [m]")
plt.grid(True)
fig.tight_layout()
plt.title('vertex distribution')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'simOutputVertexXY.pdf'))
plt.clf()

fig, ax = plt.subplots(1, 1)
plt.hist2d(azimuths / units.deg, zeniths / units.deg, bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap=plt.get_cmap('Blues'))
cb = plt.colorbar()
cb.set_label("number of events")
ax.set_aspect('equal')
plt.xlabel("azimuth [deg]")
plt.ylabel("zenith [deg]")
plt.grid(True)
plt.title('direction distribution')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'simOutputDirection.pdf'))
plt.clf()

plt.subplots(figsize = (12.8, 9.6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.hist(max_amp_ray_solution[:, i, 0], log = True)
    plt.xlabel("amp({}, {}) [V]".format(int(i % 4), int(i / 4)))
plt.suptitle("max_amp_ray_solution")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'amp.pdf'))
plt.clf()

plt.subplots(figsize = (12.8, 9.6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.hist(travel_times[:, i, 0], log = False)
    plt.xlabel("time({}, {}) [ns]".format(int(i % 4), int(i / 4)))
plt.suptitle("travel_times")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'time.pdf'))
plt.clf()

data = {"time(0, 1)": travel_times[:, 1, 0],
        "time(1, 0)": travel_times[:, 4, 0],
        "amp(0, 1)": max_amp_ray_solution[:, 1, 0],
        "amp(1, 0)": max_amp_ray_solution[:, 4, 0],
        "rr": rr,
        "tt": tt,
        "pp": pp,
        "az": azimuths,
        "ze": zeniths,
       }
df = pd.DataFrame(data, columns = ["time(0, 1)", "time(1, 0)", "amp(0, 1)", "amp(1, 0)", "rr", "tt", "pp", "az", "ze"])

plt.subplots(figsize = (6.4, 4.8))
corrMatrix = df.corr()
sbn.heatmap(corrMatrix, vmin = -1., vmax = 1., cmap = cm.RdBu)
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'corr.pdf'))
plt.clf()

pd.plotting.scatter_matrix(df[:100], figsize = (12.8, 9.6), alpha = 0.6, diagonal = "hist")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'scat.pdf'))
plt.clf()
