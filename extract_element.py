# A helper file used by the shell scripts in mm_bench. It is used to extract GPU indices from SLURM info.

import argparse

def extract_element_from_list(input_str, index):
    elements = []
    for element in input_str.split(","):
        if "-" in element:
            range_start, range_end = map(int, element.split("-"))
            elements += range(range_start, range_end + 1)
        else:
            elements.append(int(element))
    return elements[index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract an element from a comma-separated list of integers and integer ranges.")
    parser.add_argument("input_str", type=str, help="The input string to extract elements from.")
    parser.add_argument("index", type=int, help="The index of the element to extract.")
    args = parser.parse_args()

    result = extract_element_from_list(args.input_str, args.index)
    print(result)