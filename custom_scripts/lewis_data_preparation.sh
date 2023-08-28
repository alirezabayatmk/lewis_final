#!/bin/bash

echo "Starting data preparation script..."

mkdir downloaded_data
echo "Created directory 'downloaded_data'"

cd downloaded_data || { echo "Failed to change directory."; exit 1; }
echo "Changed directory to 'downloaded_data'"

mkdir cmv
echo "Created directory 'cmv'"

cd cmv || { echo "Failed to change directory."; exit 1; }
echo "Changed directory to 'cmv'"

mkdir dm1-app
echo "Created directory 'dm1-app'"

mkdir dm2-inapp
echo "Created directory 'dm2-inapp'"

python3 data_preparation.py
echo "Ran data_preparation.py"

echo "Data preparation script completed successfully."
