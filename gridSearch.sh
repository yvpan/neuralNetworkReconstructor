#!/bin/sh

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do
        for e in 50 100 200 500
        do
            for b in 32 64 128 256
            do
                #python nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/1e16.5_n1e5_ARA02_*.hdf5.XFDTD train ${l} ${n} ${e} ${b} &
                #python nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD train ${l} ${n} ${e} ${b} &
                echo "python nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD train ${l} ${n} ${e} ${b} &"
            done
        done
    done
done
wait
