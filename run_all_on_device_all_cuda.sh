#!/bin/bash


# Example: bash ./run_all_on_device_all_cuda.sh /shapes/public/example.csv ~/bob/output_files 0

in_file=$1
out_folder=$2
gpu_device_idx=$3

bash ./run_all_on_device.sh 11.4 $in_file $out_folder $gpu_device_idx
bash ./run_all_on_device.sh 11.8 $in_file $out_folder $gpu_device_idx
bash ./run_all_on_device.sh 12.0 $in_file $out_folder $gpu_device_idx
