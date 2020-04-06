#!/usr/bin/env python3

import sys

"""
Based on supervised_closed_sequence_miner, returns only closed patterns
"""

def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])
    # TODO: read the dataset files and call your miner to print the top k itemsets


if __name__ == "__main__":
    main()
