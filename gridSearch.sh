#!/bin/sh

for l in 1 3 5 7
do
    for n in 8 16 32 64
    do
        for e in 50 100 200 500
        do
            for b in 32 64 128 256
            do
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD train ${l} ${n} ${e} ${b} 0 &
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
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD rr ${l} ${n} ${e} ${b} 0 &
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
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD tt ${l} ${n} ${e} ${b} 0 &
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
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD cos ${l} ${n} ${e} ${b} 0 &
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
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD ze ${l} ${n} ${e} ${b} 0 &
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
                python ./bin/nnRecon.py /data/user/ypan/bin/simulations/ARA02Recon/*_ARA02_*.hdf5.XFDTD cosAz ${l} ${n} ${e} ${b} 0 &
            done
        done
    done
done
wait
