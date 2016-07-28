#!/bin/bash

cd data
echo "Downloading TUW model database..."
output=$(wget -c https://repo.acin.tuwien.ac.at/tmp/permanent/data/TUW_models.tar.gz)
if [ $? -ne 0 ]; then
    echo "Error downloading file"
else
    echo "File has been downloaded"
    echo "Inflating file..."
    mkdir TUW
    cd TUW

    if ! tar -zxvf ../TUW_models.tar.gz &> /dev/null; then
        echo "Failure during inflating.."
    else
        echo "Successfully inflated file! Deleting tar file..."
        cd ..
        rm TUW_models.tar.gz
        echo "Done!"
    fi
fi


echo "Downloading TUW test set..."
output=$(wget -c https://repo.acin.tuwien.ac.at/tmp/permanent/data/TUW_test_set.tar.gz)
if [ $? -ne 0 ]; then
    echo "Error downloading file"
else
    echo "File has been downloaded"
    echo "Inflating file..."
    mkdir -p TUW/test_set
    cd TUW/test_set

    if ! tar -zxvf ../../TUW_test_set.tar.gz &> /dev/null; then
        echo "Failure during inflating.."
    else
        echo "Successfully inflated file! Deleting tar file..."
        cd ../..
        rm TUW_test_set.tar.gz
        echo "Done!"
    fi
fi



echo "Downloading ground-truth labels..."
output=$(wget -c https://repo.acin.tuwien.ac.at/tmp/permanent/data/TUW_annotations.tar.gz)
if [ $? -ne 0 ]; then
    echo "Error downloading file"
else
    echo "File has been downloaded"
    echo "Inflating file..."
    mkdir -p TUW/annotations
    cd TUW/annotations

    if ! tar -zxvf ../../TUW_annotations.tar.gz &> /dev/null; then
        echo "Failure during inflating.."
    else
        echo "Successfully inflated file! Deleting tar file..."
        cd ../..
        rm TUW_annotations.tar.gz
        echo "Done!"
    fi
fi

cd ..
