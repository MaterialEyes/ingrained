sim_dir=ingrained/simulation/
dm3_dir=test/dm3/
sxm_dir=test/sxm/
structure_dir=test/structures/

sim_exec_1=ingrained/simulation/incostem-mac
if [ ! -f "$sim_exec_1" ]; then
    if [ ! -d "$sim_dir" ] ; then
        mkdir ingrained/simulation
    fi
    gdown -O ingrained/simulation/incostem-mac https://drive.google.com/uc?id=1Seb3bGwUigf_GXIeJwrlQ5FzYKuAdWKH;
    cd ingrained/simulation;
    chmod +x incostem-mac ;
    cd ../..
else
    echo "Already downloaded incostem-mac"
fi

sim_exec_2=ingrained/simulation/incostem-linux
if [ ! -f "$sim_exec_2" ]; then
    if [ ! -d "$sim_dir" ] ; then
        mkdir ingrained/simulation
    fi
    gdown -O ingrained/simulation/incostem-linux https://drive.google.com/uc?id=1aqZ4SU7o7ytpLdva7vIIPtLhJcvPj8Hl;
    cd ingrained/simulation;
    chmod +x incostem-linux ;
    cd ../..
else
    echo "Already downloaded incostem-linux"
fi

dm3_test1=test/dm3/0149.dm3
if [ ! -f "$dm3_test1" ]; then
    if [ ! -d "$dm3_dir1" ] ; then
        mkdir test/dm3
    fi
    gdown -O test/dm3/0149.dm3 https://drive.google.com/uc?id=1O3c5xWOv-EoSvtZhOdB6oeyKhpGWVSlp
else
    echo "Already downloaded test/dm3/0149.dm3"
fi

dm3_test2=test/dm3/039.dm3
if [ ! -f "$dm3_test2" ]; then
    if [ ! -d "$dm3_dir2" ] ; then
        mkdir test/dm3
    fi
    gdown -O test/dm3/039.dm3 https://drive.google.com/uc?id=1x6MCdPmS-wpPOT0IriYhtXCuDR8iDagn
else
    echo "Already downloaded test/dm3/039.dm3"
fi

sxm_test1=test/sxm/0729LHe_D_BAgM14056.sxm
if [ ! -f "$sxm_test1" ]; then
    if [ ! -d "$sxm_dir1" ] ; then
        mkdir test/sxm
    fi
    gdown -O test/sxm/0729LHe_D_BAgM14056.sxm https://drive.google.com/uc?id=1V5t5cIgjJrH9kxm2zN9zKcJycOzHi4Wr
else
    echo "Already downloaded test/sxm/0729LHe_D_BAgM14056.sxm"
fi

sxm_test2=test/sxm/0729LHe_D_BAgM14064.sxm
if [ ! -f "$sxm_test2" ]; then
    if [ ! -d "$sxm_dir2" ] ; then
        mkdir test/sxm
    fi
    gdown -O test/sxm/0729LHe_D_BAgM14064.sxm https://drive.google.com/uc?id=1GefKSSH1oY0_Qjhx5TfTrg6NENaayXdy
else
    echo "Already downloaded test/sxm/0729LHe_D_BAgM14064.sxm"
fi

structure_test=test/structures/0149.POSCAR.vasp
if [ ! -f "$structure_test" ]; then
    if [ ! -d "$structure_dir" ] ; then
        mkdir test/structures
    fi
    gdown -O test/structures/0149.POSCAR.vasp https://drive.google.com/uc?id=1MT2t22lAKUlf4lm2PhaeU99gwDnwhDHC
else
    echo "Already downloaded test/structures/0149.POSCAR.vasp"
fi

structure_test_pchg_1=test/structures/PARCHG_4
if [ ! -f "$structure_test_pchg_1" ]; then
    if [ ! -d "$structure_dir" ] ; then
        mkdir test/structures
    fi
    gdown -O test/structures/PARCHG_4 https://drive.google.com/uc?id=1XgrlexXUVTOxJYfl-OW278TWJfpHCyAb
else
    echo "Already downloaded test/structures/PARCHG_4"
fi

structure_test_pchg_2=test/structures/PARCHG
if [ ! -f "$structure_test_pchg_2" ]; then
    if [ ! -d "$structure_dir" ] ; then
        mkdir test/structures
    fi
    gdown -O test/structures/PARCHG https://drive.google.com/uc?id=1JElWFbl1vsV7a2ceDxg6HVqyjVHfGIhM
else
    echo "Already downloaded test/structures/PARCHG"
fi

git clone https://bitbucket.org/piraynal/pydm3reader/;
cd pydm3reader/;
pip install .;
cd ..;
rm -rf pydm3reader;
