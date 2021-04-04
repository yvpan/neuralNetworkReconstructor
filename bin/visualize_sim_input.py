import numpy as np
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
import h5py
import argparse
import os

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list input')
parser.add_argument('inputfilename', type=str, nargs = '+', help='path to NuRadioMC hdf5 simulation input')
args = parser.parse_args()
filename = os.path.splitext(os.path.basename(args.inputfilename[0]))[0]
dirname = os.path.dirname(args.inputfilename[0])
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)
fin = h5py.File(args.inputfilename[0], 'r')
print("Reading " + str(args.inputfilename[0]))
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])
zeniths = np.array(fin['zeniths'])
azimuths = np.array(fin['azimuths'])

for i in range(len((args.inputfilename)) - 1):
    fin = h5py.File(args.inputfilename[i + 1], 'r')
    print("Reading " + str(args.inputfilename[i + 1]))
    xx = np.append(xx, np.array(fin['xx']))
    yy = np.append(yy, np.array(fin['yy']))
    zz = np.append(zz, np.array(fin['zz']))
    zeniths = np.append(zeniths, np.array(fin['zeniths']))
    azimuths = np.append(azimuths, np.array(fin['azimuths']))

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
plt.savefig(os.path.join(plot_folder, 'simInputVertexRZ.pdf'), bbox_inches = "tight")
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
plt.savefig(os.path.join(plot_folder, 'simInputVertexXY.pdf'), bbox_inches = "tight")
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
plt.savefig(os.path.join(plot_folder, 'simInputDirection.pdf'), bbox_inches = "tight")
plt.clf()
