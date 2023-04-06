import argparse
import sys

from common_types import KDecomposition, TritonMMKernelStyle

def parse_args(
    k_decomposition_choices, 
    kernel_style_choices,
) -> argparse.Namespace:
    
    k_decomposition_choices = [x.name for x in k_decomposition_choices]
    kernel_style_choices = [x.name for x in kernel_style_choices]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file",
        type=str,
        required=True,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=False,
        help="Output CSV file. If not specified, write to stdout.",
        default=None,
    )
    parser.add_argument(
        "--trial_num",
        type=int,
        required=False,
        help="Trial number.",
        default=1,
    )
    parser.add_argument(
        "--profiling_iterations",
        type=int,
        required=False,
        help="Number of iterations to profile when profiling a kernel.",
        default=20,
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        required=False,
        help="Number of iterations to warmup when profiling a kernel.",
        default=10,
    )
    parser.add_argument(
        "--k_decomposition",
        type=str,
        required=len(k_decomposition_choices) > 1,
        help="k decomposition",
        default=k_decomposition_choices[0],
        choices=k_decomposition_choices,
    )
    parser.add_argument(
        "--kernel_style",
        type=str,
        required=len(kernel_style_choices) > 1,
        help="kernel style",
        default=kernel_style_choices[0],
        choices=kernel_style_choices,
    )
    parser.add_argument(
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Flag to enable dry_run, which forces warmup_iterations and profiling_iterations to 1.",
    )

    parsed_args = parser.parse_args()
    
    if parsed_args.out_file is not None:
        sys.stdout = open(parsed_args.out_file, 'w')

    if parsed_args.dry_run:
        parsed_args.profiling_iterations = 1
        parsed_args.warmup_iterations = 1

    parsed_args.k_decomposition = KDecomposition[parsed_args.k_decomposition]
    parsed_args.kernel_style = TritonMMKernelStyle[parsed_args.kernel_style]

    return parsed_args
