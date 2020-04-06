#!/usr/bin/env python3

import sys

"""
Algo based on Weighted relative accuracy

Finds the k best paterns that are highly present in the positive class, but not in the negative class.

=> consider Wracc score
"""

def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])
    # TODO: read the dataset files and call your miner to print the top k itemsets


if __name__ == "__main__":
    main()
