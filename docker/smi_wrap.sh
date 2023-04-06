#!/bin/bash

# This script provides an alias to run commands with to ensure they are being done with pinned clocks.

cuda_v="$CUDA_VERSION"
silence_smi_out="1"


if [ "$silence_smi_out" == "0" ]; then

    if [ "$cuda_v" == "11.4.0" ] || [ "$cuda_v" == "12.0.1" ]; then
        cmd1="sudo nvidia-smi -i 0 -pm 1; sudo nvidia-smi --power-limit=330 -i 0; sudo nvidia-smi -lgc 1410 -i 0;"
    elif [ "$cuda_v" == "11.8.0" ]; then
        cmd1="nvidia-smi -i 0 -pm 1; nvidia-smi --power-limit=330 -i 0; nvidia-smi -lgc 1410 -i 0;"
    fi

    if [ "$cuda_v" == "11.4.0" ] || [ "$cuda_v" == "12.0.1" ]; then
        cmd3="sudo nvidia-smi -rgc -i 0; sudo nvidia-smi --power-limit=400 -i 0;"
    elif [ "$cuda_v" == "11.8.0" ]; then
        cmd3="nvidia-smi -rgc -i 0; nvidia-smi --power-limit=400 -i 0;"
    fi

else

    if [ "$cuda_v" == "11.4.0" ] || [ "$cuda_v" == "12.0.1" ]; then
        cmd1="sudo nvidia-smi -i 0 -pm 1 >/dev/null 2>&1; sudo nvidia-smi --power-limit=330 -i 0 >/dev/null 2>&1; sudo nvidia-smi -lgc 1410 -i 0 >/dev/null 2>&1;"
    elif [ "$cuda_v" == "11.8.0" ]; then
        cmd1="nvidia-smi -i 0 -pm 1 >/dev/null 2>&1; nvidia-smi --power-limit=330 -i 0 >/dev/null 2>&1; nvidia-smi -lgc 1410 -i 0 >/dev/null 2>&1;"
    fi

    if [ "$cuda_v" == "11.4.0" ] || [ "$cuda_v" == "12.0.1" ]; then
        cmd3="sudo nvidia-smi -rgc -i 0 >/dev/null 2>&1; sudo nvidia-smi --power-limit=400 -i 0 >/dev/null 2>&1;"
    elif [ "$cuda_v" == "11.8.0" ]; then
        cmd3="nvidia-smi -rgc -i 0 >/dev/null 2>&1; nvidia-smi --power-limit=400 -i 0 >/dev/null 2>&1;"
    fi

fi


# Get the arbitrary command
input_cmd="$@"

# Run cmd1
eval "$cmd1"

# Run the input command
eval "$input_cmd"

# Run cmd3
eval "$cmd3"