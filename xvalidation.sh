#!/bin/sh
'''
for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD train 5 16 200 64 ${f} &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD rr 5 16 200 64 ${f} &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD tt 5 16 200 64 ${f} &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD cos 5 16 200 64 ${f} &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD ze 5 16 200 64 ${f} &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD cosAz 5 16 200 64 ${f} &
done
wait
'''
for f in {0..4}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_0.hdf5.XFDTD train 3 8 50 64 ${f} &
done
wait
