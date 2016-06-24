#!/bin/bash

cd data
echo "Downloading Cat10 model database from 3dNet..."
output=$(wget -c https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/Cat10_ModelDatabase.zip)
if [ $? -ne 0 ]; then
    echo "Error downloading file"
else
    echo "File has been downloaded"
    echo "Unzipping..."
    mkdir 3dNet
    cd 3dNet

    if ! unzip ../Cat10_ModelDatabase.zip &> /dev/null; then
        echo "Failure during unzipping.."
    else
        echo "Successfully unzipped file! Cleaning up..."

        #remove broken .ply-files (they produce segmentation fault during rendering - do not know why)
        rm Cat10_ModelDatabase/hammer/6683cadf1d45d69a3a3cbcf1ae739c36.ply
        rm Cat10_ModelDatabase/mug/a35a92dcc481a994e45e693d882ad8f.ply
        rm Cat10_ModelDatabase/mug/b88bcf33f25c6cb15b4f129f868dedb.ply
        rm Cat10_ModelDatabase/tetra_pak/a28d27ec2288b962f22e103132f5d39e.ply
        rm Cat10_ModelDatabase/toilet_paper/da6e2afe4689d66170acfa4e32a0a86.ply

        cd ..
        rm Cat10_ModelDatabase.zip
        echo "Done!"
    fi
fi

cd ..

