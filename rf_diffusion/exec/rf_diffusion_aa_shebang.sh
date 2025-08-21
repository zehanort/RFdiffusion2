#!/usr/bin/bash

###################
# You can add the path to this file as the shebang line in your python script. 
# Then by default, the python script will be executed with the python interpreter
# in the SIF_PATH container. Here, we launch the container with nvidia gpu and slurm support.
#
# Example shebang: #!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
###################

maybe_echo() {
  if [[ "$RFD2_DEBUG" == "1" ]]; then
    echo "$@"
  fi
}

# Let the user know this script is setting things up behind the scene
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
maybe_echo '################## Start shebang info ##################'
maybe_echo "The file $SCRIPT_PATH is being run as a shebang executable. It will...
    1. Add the "rf_diffusion" repo directory to your PYTHONPATH.
    2. Run your python script from the right container, which contains all dependencies.
    3. Launch the container with slurm and nvidia gpu support."

# Extract the path to the Python script from the arguments
PYTHON_SCRIPT=$(realpath "$1")
shift

# Automatically add the git repo dir to the PYTHONPATH
PKG_NAME=rf_diffusion
if [[ $PYTHON_SCRIPT =~ $PKG_NAME ]]
then
    PKG_DIR=$(echo "$PYTHON_SCRIPT" | sed -E 's/^(.*\/'$PKG_NAME'\/).*/\1/')
    REPO_DIR=$(dirname "$PKG_DIR")
    
    if [[ $PYTHONPATH =~ $REPO_DIR ]]
    then
        maybe_echo "The repo dir ($REPO_DIR) is already in the PYTHONPATH. PYTHONPATH will remain as $PYTHONPATH"
    else
        export PYTHONPATH=$PYTHONPATH:$REPO_DIR
        maybe_echo "The repo dir ($REPO_DIR) was not in the PYTHONPATH. PYTHONPATH is now $PYTHONPATH"
    fi

else
    echo "The script $PYTHON_SCRIPT is not in the \"rf_diffusion\" package. Are you sure sure you're using the right shebang?"
    exit 1
fi

# check if we are at the IPD
IPD_FILE="/software/containers/versions/rf_diffusion_aa/ipd.txt"

SIF_PATH=""

if [ -z "$APPTAINER_NAME" ]; then

    # This is the default apptainer that you can build from exec/rf_diffusion_aa.spec
    SIF_PATH="$SCRIPT_DIR/rf_diffusion_aa.sif"

    if [ ! -f $SIF_PATH ]; then
        echo "Default apptainer not found (you can build it from exec/rf_diffusion_aa.spec): $SIF_PATH"
        SIF_PATH=""

        if [ -f $IPD_FILE ]; then
            SIF_PATH=$(readlink -f "$SCRIPT_DIR/bakerlab_rf_diffusion_aa.sif" )
            if [ -z $SIF_PATH ] || [ ! -f $SIF_PATH ]; then
                SIF_PATH=""
                echo "You're at the IPD and something is wrong. The target of this symlink doesn't exist: $SCRIPT_DIR/bakerlab_rf_diffusion_aa.sif"
            else
                echo "You're at the IPD and we found this sif: $SIF_PATH"
            fi
        fi

        if [ -z $SIF_PATH ]; then
            echo "No apptainer found. Attempting to run $PYTHON_SCRIPT with $(which python)"
        fi
    fi
else
    maybe_echo "Already running inside container $APPTAINER_NAME. Executing $PYTHON_SCRIPT with $(which python) in the existing container."
fi

if [ ! -z $SIF_PATH ]; then
    maybe_echo "Running $PYTHON_SCRIPT with $SIF_PATH."
    maybe_echo '################## End shebang info ####################'
    maybe_echo
    /usr/bin/apptainer run --nv --slurm --env PYTHONPATH="\$PYTHONPATH:$PYTHONPATH" $SIF_PATH "$PYTHON_SCRIPT" "$@"
else
    maybe_echo '################## End shebang info ####################'
    maybe_echo
    python "$PYTHON_SCRIPT" "$@"
fi