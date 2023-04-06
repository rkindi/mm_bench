# mm_bench
Runs benchmark of cuBLAS vs Triton vs CUTLASS over a suite of problem sizes. Uses docker to run the benchmarks across various environments such as CUDA 11.4, 11.8. 12.0.

## Quick Start

The instructions below let you evaluate the shapes in shapes/public/mini_example.csv across all CUDA / backend combinations using the AWS cluster.

1. `tmux` on the jump host. Run the commands below within tmux on the jump host. The reason for this is to have the `salloc` to continue living even when your VSCode disconnects or you quit VSCode. If you don't do this, `salloc` will kill the job if you get disconnected or quit VSCode.
2. `salloc -N 1 -p dev --cpus-per-task=16 -t 5:00:00 --gpus-per-node=1` We use `salloc` because sbatch is having some permission issues with docker.
3. `ssh` into the `salloc`ed node.
4. `cd` to the `mm_bench` folder (where this README.md file is located) as a lot of the scripts in the repo use relative paths.
5. Run `bash ./run_all_on_device_all_cuda.sh /shapes/public/mini_example.csv ~/bob/output_files 0`. This command runs the shapes in mini_example.csv on all the backends (stable & legacy Triton, CUBLAS, CUTLASS). It stores the output files in `~/bob/output_files`. Note `/shapes/public/mini_example.csv` is suggesting the `/shapes` folder lives in the root -- that's because this path is referring to where `./shapes/` in this repo is copied to the docker container. The `0` argument tells docker which GPU to run. Currently slurm and docker don't play nice where docker can see all of the GPUs on the host even when the slurm allocation gives you a subset of the GPUs. The scripts in this repo use this device index argument to use the correct GPU of the ones slurm gives us so we can be respectful of our neighbors on the cluster.
6. Note, if you want to reserve more GPUs on the host and process multiple shape files at the same time, you could do something like:
`salloc -N 1 -p dev --cpus-per-task=16 -t 5:00:00 --gpus-per-node=1` and `bash ./run_all_on_device_all_cuda.sh /shapes/public/mini_example.csv ~/bob/output_files 0` `bash ./run_all_on_device_all_cuda.sh /shapes/public/some_other_shapes.csv ~/bob/output_files 1`. The latter 2 commands will run the benchmarks for mini_example.csv on GPU 0 and the benchmarks for some_other_shapes.csv on GPU 1.


## Notes

- The supported Triton modes are legacy (triton-legacy git submodule) and stable (uses triton-torch-inductor-stable git submodule which maps to stable version of Triton MLIR backend). There's also modes pt_nightly and mlir modes which map to the Triton version used by PyTorch nightly and trunk Triton MLIR, but those are not tested, so it is suggested not to use those. If you use `./run_all_on_device_all_cuda.sh`, you'll be using the legacy and stable modes.