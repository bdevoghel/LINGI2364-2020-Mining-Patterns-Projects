#!/usr/bin/env python3

import sys
from copy import copy, deepcopy
from collections import OrderedDict

"""
Algo based on SPADE or PrefixSpan

Finds the k most frequent paterns in both + and - classes of the dataset.

=> consider sum of of the support of both classes
"""

POS = 1
NEG = -1

class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan"):
        """reads the dataset files and initializes the dataset"""

        try:
            if algo == "PrefixSpan":
                self.algo = algo
                self.transactions = OrderedDict()
                last_pos = sys.maxsize
                tid = 0
                for path_id, filepath in enumerate([filepath_positive, filepath_negative]):
                    path_id = POS if path_id == 0 else NEG

                    lines = [line.strip() for line in open(filepath, "r")]
                    lines = [line.split(" ") for line in lines if line]  # skipping blank lines

                    for line in lines:
                        symbol, pos = line
                        pos = int(pos)
                        if pos < last_pos:
                            tid += 1
                            self.transactions[tid] = {'symbols':[], 'pid':0, 'class':path_id}
                        self.transactions[tid]['symbols'].append(symbol)
                        last_pos = pos

                self.projection = []

        except IOError as e:
            print("Unable to read file !\n" + e)
    
    def __repr__(self):
        return str(self.transactions).replace('(', '\n(')


def prefixspan(dataset, k):
    queue = [dataset] # add:append(ELEM), remove:pop(0)
    while queue and k > 0:
        for transaction in queue.pop(0).transactions.items():
            

           


def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])

    # read the dataset files 
    dataset = Dataset(pos_filepath, neg_filepath, algo="PrefixSpan")
    prefixspan(dataset, k)


if __name__ == "__main__":
    main()
