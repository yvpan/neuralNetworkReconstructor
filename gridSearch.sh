#!/bin/sh

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do
        for e in 50 100 200 500
        do
            for b in 32 64 128 256
            do
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ train 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do  
        for e in 50 100 200 500
        do  
            for b in 32 64 128 256
            do  
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ rr 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do  
        for e in 50 100 200 500
        do  
            for b in 32 64 128 256
            do  
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ tt 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do  
        for e in 50 100 200 500
        do  
            for b in 32 64 128 256
            do  
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ pp 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do  
        for e in 50 100 200 500
        do  
            for b in 32 64 128 256
            do  
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ ze 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do  
        for e in 50 100 200 500
        do  
            for b in 32 64 128 256
            do  
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ az 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do
        for e in 50 100 200 500
        do
            for b in 32 64 128 256
            do
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_0.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy /data/user/ypan/bin/simulations/ARA02Recon/data/*_ARA02_1.hdf5.XFDTD*.nur_nurNoisePowerHilbert.npy ./plots/nnRecon/ sh 1 ${l} ${n} ${e} ${b}  0 0 0 &
            done
        done
    done
done
wait
