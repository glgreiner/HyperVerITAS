#!/usr/bin/env python3

import argparse
import subprocess
from scripts.util import *
import time

def verify_tile_proof(circuit_name):

    print('\n====================================')
    print(f'Verifying proof for {circuit_name}')

    for name,command in {'VERIFY':f'time -v ./scripts/proving_system/verifier.sh {circuit_name}'}.items():
        start = time.time()
        res = subprocess.run(command, shell=True, universal_newlines=True)
        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")
        if res.returncode != 0:
            raise Exception(f'Error in command: {command}\n{res.stderr}')
        else:
            print(f'Command {name} executed successfully')
    
    print('\n====================================')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Generate proof for an image circuit.')

    # parser.add_argument('--image', type=str, required=True, help='Path to the image.')
    # parser.add_argument('--N', type=int, required=True, help='Number of tiles to split the image into.')
    # parser.add_argument('--height', type=int, required=True, help='Resize height.')
    # parser.add_argument('--width', type=int, required=True, help='Resize width.')
    # parser.add_argument('--pot', type=str, required=True, help='Path to the pot file.')

    # args = parser.parse_args()

    # generate_proof(args.image, args.N, args.height, args.width, args.pot)

    parser = argparse.ArgumentParser(description='Verify proof for an image circuit.')

    parser.add_argument('--circuit', type=str, required=True, help='circuit name.')

    args = parser.parse_args()

    verify_tile_proof(args.circuit)