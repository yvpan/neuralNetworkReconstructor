#!/bin/sh

#python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD train 5 16 200 64 0
python ./bin/nnRecon.py ./data/tong_1e18.5_n1e6_1015_v121.part0000.hdf5_out.hdf5 train 5 16 200 64 0
