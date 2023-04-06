"""
Small helper script that takes a shape file as input and then emits another shape file
with each shape value padded.
"""

import argparse
from io_utils import CSVProblemReader, ProblemSpec

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file",
        type=str,
        required=True,
        help="Input shapes file.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output shapes file.",
        default=None,
    )
    parser.add_argument(
        "--multiple",
        type=int,
        required=True,
        help="All shape values will be changed to be multiples of this value.",
        default=1,
    )

    parsed_args = parser.parse_args()
    return parsed_args

def pad(num, multiple):
    return ((num + multiple - 1) // multiple) * multiple

def pad_shapes(shapes, multiple):
    new_shapes = [
        ProblemSpec(
            shape.op_type,
            shape.bias_type,
            shape.data_type,
            shape.b,
            pad(shape.m, multiple),
            pad(shape.n, multiple),
            pad(shape.k, multiple),
            shape.layout,
        )
        for shape in shapes
    ]
    return new_shapes

def write_shapes_to_file(fname, shapes):
    with open(fname, "w") as f:
        for shape in shapes:
            f.write(
                ",".join(
                    map(
                        str,
                        [
                            shape.op_type.name,
                            shape.bias_type.name,
                            shape.b,
                            shape.m,
                            shape.n,
                            shape.k,
                            shape.layout.name,
                        ],
                    )
                )
            )
            f.write("\n")

def main():
    parsed_args = parse_args()
    shapes = CSVProblemReader.read(parsed_args.in_file)
    new_shapes = pad_shapes(shapes, parsed_args.multiple)
    write_shapes_to_file(parsed_args.out_file, new_shapes)


if __name__ == "__main__":
    main()