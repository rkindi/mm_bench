cuda_version=$1
triton_option=${2:-pt_nightly}

if [ "$cuda_version" == '11.4' ]
then
    echo "Building mm_bench docker image with tag mm_bench_11_4_0_$triton_option:latest"
    DOCKER_BUILDKIT=1 docker build -f ./docker/Dockerfile.11_4_0 --build-arg MM_BENCH_TRITON_OPTION="$triton_option" -t mm_bench_11_4_0_$triton_option .
elif [ "$cuda_version" == '11.8' ]
then
    echo "Building mm_bench docker image with tag mm_bench_11_8_0_$triton_option:latest"
    DOCKER_BUILDKIT=1 docker build -f ./docker/Dockerfile.11_8_0 --build-arg MM_BENCH_TRITON_OPTION="$triton_option" -t mm_bench_11_8_0_$triton_option .
elif [ "$cuda_version" == '12.0' ]
then
    echo "Building mm_bench docker image with tag mm_bench_12_0_1_$triton_option:latest"
    DOCKER_BUILDKIT=1 docker build -f ./docker/Dockerfile.12_0_1 --build-arg MM_BENCH_TRITON_OPTION="$triton_option" -t mm_bench_12_0_1_$triton_option .
fi