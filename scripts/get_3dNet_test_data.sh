#!/bin/bash

cd data
echo "Downloading test database from 3dNet..."
output=$(wget -c https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/Cat10_TestDatabase.zip)
if [ $? -ne 0 ]; then
    echo "Error downloading file"
else
    echo "File has been downloaded"
    echo "Unzipping..."
    mkdir -p 3dNet/Cat10_TestDatabase
    cd 3dNet/Cat10_TestDatabase

    if ! unzip ../../Cat10_TestDatabase.zip &> /dev/null; then
        echo "Failure during unzipping.."
    else
        echo "Successfully unzipped file! Cleaning up..."
        
        #remove broken files
        cd pcd_binary
        mv tt_hmrr107.pcd tt_hmrr107.pcd_broken
        mv tt_hmrr116.pcd tt_hmrr116.pcd_broken
        mv tt_hmrr132.pcd tt_hmrr132.pcd_broken
        mv tt_hmrr141.pcd tt_hmrr141.pcd_broken
        cd ..
        
        cd ../..
        rm Cat10_TestDatabase.zip
        echo "Done!"
    fi
fi

cd ..

