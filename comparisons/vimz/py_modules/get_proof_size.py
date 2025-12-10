import os
import re
import json


def format_proof_file(input_file):
    output_file = os.path.splitext(input_file)[0] + "_lines.json"
    # Read one-line JSON from the input file
    with open(input_file, 'r') as infile:
        data = json.loads(infile.read())

    # Write nicely formatted JSON to the output file
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)

    print(f"Formatted JSON saved to '{output_file}'!")
    return output_file

def determine_proof_size(filename):
    count_64_len_quoted_strings = 0
    count_1to3digit_num_optional_comma = 0

    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()

            # Check for 64 non-whitespace chars inside double quotes
            if len(line) > 4 and re.search(r'\"[^\s]{64}\"', line):
                count_64_len_quoted_strings += 1
            # Check for 1-3 digit numbers followed by an optional comma
            elif len(line) <= 4 and re.search(r'\b\d{1,3},?', line): 
                count_1to3digit_num_optional_comma += 1


    num_hex = count_64_len_quoted_strings
    num_int = count_1to3digit_num_optional_comma

    print(f"\nTotal number of 64-digit hex strings: {num_hex}")
    print(f"Total number of integers in [0,255]: {num_int}")

    bytes_hex = num_hex * 32
    bytes_int = num_int
    bytes_total = bytes_hex + bytes_int 

    print(f"\nTotal Bytes of the hex strings: {bytes_hex}")
    print(f"Total Bytes of the integers: {bytes_int}")
    print(f"Total Bytes of proof: {bytes_total}")

    kb_total = float(bytes_total/1000)

    print(f"\nProof Size (in KB): {kb_total}")
  

def format_and_determine():
    filename = input("Provide the path to the json proof file:\n")

    # format it as needed
    out_file = format_proof_file(filename)

    # determine the actual proof size
    determine_proof_size(out_file)

format_and_determine()