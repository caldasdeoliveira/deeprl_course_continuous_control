# This script is designed to work on MAC 
# (only tested on Monterey version 12.2.1 with an intel i5)

# This repo uses git submodules and therefore should be cloned with the --recursive flag
# in case you forgot to do this the following line of code handles this for you
git submodule update --init

# Automated error correction in Value-based-methods repo. Torch is set to 0.4.0 which causes an error
sed -ie "s/torch==([0-9]+\.[0-9]+\.[0-9]+)/torch==1.4.0/g" Value-based-methods/python/requirements.txt

# Check if conda environment already exists and create or update it
if { conda env list | grep 'RUN_ENV'; } >/dev/null 2>&1; then
    conda env update --file env.yml --prune
    echo "Environment already exists"
else
    conda env create -f env.yml
    echo "Environment doesnt exist"
fi

conda activate drl_navigation

# locally install the unity environment
cd Value-based-methods/python
pip install .
cd ../..

# Create a kernel with the new conda environment
python -m ipykernel install --user --name drl_navigation --display-name "drl_navigation"

# Download and unzip the Unity environment
curl -O  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
unzip Reacher.app.zip
rm Reacher.app.zip