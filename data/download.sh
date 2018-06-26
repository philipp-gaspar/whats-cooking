#!/bin/bash

if [ ! -f train.json ]; then
    if kaggle -v; then
	echo "Downloading the dataset:"
    	kaggle competitions download -c whats-cooking -p $PWD

	# Unzip the downloaded files
	echo "Unzip files:"
	unzip sample_submission.csv.zip
	unzip test.json.zip
	unzip train.json.zip

	# Remove zip files
	rm *.zip
	
	printf "\nThe dataset is ready for use!\n"
    else
	echo "Please install the Kaggle API: https://github.com/Kaggle/kaggle-api"
    fi

else
    echo "The dataset has already been downloaded."
fi
