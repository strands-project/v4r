#!/bin/bash

cd data
echo "Downloading test database from 3dNet..."
output=$(wget -c https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/Cat10_TestDatabase.zip)
if [ $? -ne 0 ]; then
    echo "Error downloading file"
else
    echo "File has been downloaded"
    echo "Unzipping..."
    mkdir 3dNet
    cd 3dNet

    if ! unzip ../Cat10_TestDatabase.zip &> /dev/null; then
        echo "Failure during unzipping.."
    else
        echo "Successfully unzipped file! Cleaning up..."
        cd ..
        rm Cat10_TestDatabase.zip
        echo "Done!"
    fi
fi

cd ..

