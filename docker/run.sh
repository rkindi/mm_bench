# Simple run script which uses all CPUs and GPUs. See the various scripts in the root folder to see how to run containers with only the GPUs slurm exposes.

cuda_version=$1
triton_option=${2:-pt_nightly}

if [ "$cuda_version" == '11.4' ]
then
    docker run -it --gpus all --cap-add SYS_ADMIN mm_bench_11_4_0_$triton_option:latest
elif [ "$cuda_version" == '11.8' ]
then
    docker run -it --gpus all --cap-add SYS_ADMIN mm_bench_11_8_0_$triton_option:latest
elif [ "$cuda_version" == '12.0' ]
then
    docker run -it --gpus all --cap-add SYS_ADMIN mm_bench_12_0_1_$triton_option:latest
fi
