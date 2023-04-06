#!/bin/bash

# Example: bash ./run_all.sh 11.4 /shapes/public/example.csv ~/bob/output_files

cuda_version=$1
in_file=$2
out_folder=$3

out_file_base=$(basename $in_file .csv)


# Check if the folder path ends with a backslash
if [[ "${out_folder}" != */ ]]
then
  # Append a backslash to the folder path
  out_folder="${out_folder}/"
fi

out_file_base="/output_files/$out_file_base"

if [ "$cuda_version" == '11.4' ]
then
    cuda_version_underscores='11_4_0'
    cuda_version_nounderscores='1140'
elif [ "$cuda_version" == '11.8' ]
then
    cuda_version_underscores='11_8_0'
    cuda_version_nounderscores='1180'
elif [ "$cuda_version" == '12.0' ]
then
    cuda_version_underscores='12_0_1'
    cuda_version_nounderscores='1201'
else
    echo "Unsupported cuda version: $cuda_version"
    exit
fi

get_out_file_eager() {
    out_file_base=$1
    cuda_version_nounderscores=$2
    echo "${out_file_base}_${cuda_version_nounderscores}_eager.txt"
}

get_out_file_cutlass() {
    out_file_base=$1
    cuda_version_nounderscores=$2
    k_decomp=$3
    echo "${out_file_base}_${cuda_version_nounderscores}_cutlass_${k_decomp}.txt"
}

get_out_file_triton() {
    out_file_base=$1
    cuda_version_nounderscores=$2
    k_decomp=$3
    triton_option=$4
    kernel_style=$5
    echo "${out_file_base}_${cuda_version_nounderscores}_triton-${triton_option}-${kernel_style}_${k_decomp}.txt"
}

####################################### 11.4 LEGACY #######################################

# BUILD
triton_option='legacy'
bash docker/build.sh $cuda_version $triton_option

# START CONTAINER
my_docker_container_id=$(docker run -v ${out_folder}:/output_files -v ./cutlass/:/cutlass -v ./scripts/:/scripts -v ./shapes/:/shapes:ro -d -it --gpus "device=$(scontrol show job -d $SLURM_JOB_ID | grep -oP '(?<=GRES=gpu:1\(IDX:)[0-9]+(?=\))')" --cap-add SYS_ADMIN mm_bench_${cuda_version_underscores}_${triton_option}:latest)
# EAGER
echo Step 1
out_file=$(get_out_file_eager "$out_file_base" "$cuda_version_nounderscores")
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/eager_run.py --in_file=$in_file --out_file=$out_file

# CUTLASS
echo Step 2
out_file=$(get_out_file_cutlass "$out_file_base" "$cuda_version_nounderscores" none)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/cutlass_run.py --in_file=$in_file --k_decomposition=NONE --out_file=$out_file

echo Step 3
out_file=$(get_out_file_cutlass "$out_file_base" "$cuda_version_nounderscores" splitkserial)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/cutlass_run.py --in_file=$in_file --k_decomposition=SPLIT_K_SERIAL --out_file=$out_file

echo Step 4
out_file=$(get_out_file_cutlass "$out_file_base" "$cuda_version_nounderscores" splitkparallel)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/cutlass_run.py --in_file=$in_file --k_decomposition=SPLIT_K_PARALLEL --out_file=$out_file

echo Step 5
out_file=$(get_out_file_cutlass "$out_file_base" "$cuda_version_nounderscores" streamk)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/cutlass_run.py --in_file=$in_file --k_decomposition=STREAM_K --out_file=$out_file

# TRITON (LEGACY)
echo Step 6
out_file=$(get_out_file_triton "$out_file_base" "$cuda_version_nounderscores" none $triton_option inductor)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/triton_run.py --in_file=$in_file --k_decomposition=NONE --kernel_style=INDUCTOR --out_file=$out_file

echo Step 7
out_file=$(get_out_file_triton "$out_file_base" "$cuda_version_nounderscores" none $triton_option tritonopsmatmul)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/triton_run.py --in_file=$in_file --k_decomposition=NONE --kernel_style=TRITON_OPS_MATMUL --out_file=$out_file

echo Step 8
out_file=$(get_out_file_triton "$out_file_base" "$cuda_version_nounderscores" splitkserial $triton_option tritonopsmatmul)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/triton_run.py --in_file=$in_file --k_decomposition=SPLIT_K_SERIAL --kernel_style=TRITON_OPS_MATMUL --out_file=$out_file

# STOP CONTAINER
docker kill $my_docker_container_id

####################################### 11.4 STABLE #######################################

# BUILD
triton_option='stable'
bash docker/build.sh $cuda_version $triton_option

# START CONTAINER
my_docker_container_id=$(docker run -v ${out_folder}:/output_files -v ./cutlass/:/cutlass -v ./scripts/:/scripts -v ./shapes/:/shapes:ro -d -it --gpus "device=$(scontrol show job -d $SLURM_JOB_ID | grep -oP '(?<=GRES=gpu:1\(IDX:)[0-9]+(?=\))')" --cap-add SYS_ADMIN mm_bench_${cuda_version_underscores}_${triton_option}:latest)

# TRITON (STABLE)
echo Step 9
out_file=$(get_out_file_triton $out_file_base $cuda_version_nounderscores none $triton_option inductor)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/triton_run.py --in_file=$in_file --k_decomposition=NONE --kernel_style=INDUCTOR --out_file=$out_file

echo Step 10
out_file=$(get_out_file_triton $out_file_base $cuda_version_nounderscores none $triton_option tritonopsmatmul)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/triton_run.py --in_file=$in_file --k_decomposition=NONE --kernel_style=TRITON_OPS_MATMUL --out_file=$out_file

echo Step 11
out_file=$(get_out_file_triton $out_file_base $cuda_version_nounderscores splitkserial $triton_option tritonopsmatmul)
echo $out_file
docker exec -it $my_docker_container_id bash /smi_wrap.sh python3 scripts/triton_run.py --in_file=$in_file --k_decomposition=SPLIT_K_SERIAL --kernel_style=TRITON_OPS_MATMUL --out_file=$out_file

# STOP CONTAINER
docker kill $my_docker_container_id