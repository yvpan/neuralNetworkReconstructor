#!/bin/sh

#python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ sh 0 3 60 100 64 0 0 0
python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ sh 1 3 60 100 64 0 0 0
