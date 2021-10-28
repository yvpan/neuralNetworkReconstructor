#!/bin/sh

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ train 0 3 60 100 64 ${f} 0 0 &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ rr 0 3 60 100 64 ${f} 0 0 &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ tt 0 3 60 100 64 ${f} 0 0 &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ pp 0 3 60 100 64 ${f} 0 0 &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ ze 0 3 60 100 64 ${f} 0 0 &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ az 0 3 60 100 64 ${f} 0 0 &
done
wait

for f in {0..9}
do
    python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD ./plots/nnRecon/ sh 0 3 60 100 64 ${f} 0 0 &
done
wait
