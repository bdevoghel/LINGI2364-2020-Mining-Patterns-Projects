#!/usr/bin/env python3

import sys
from copy import deepcopy
from collections import defaultdict
from heapq import heappop, heappush, heapify


"""
Algo based on SPADE or PrefixSpan

Finds the k most frequent paterns in both + and - classes of the dataset.

=> consider sum of of the support of both classes
"""

POS = 1
NEG = -1

class Heap:

    def __init__(self, order='min'):
        self.order = -1 if order == 'max' else 1
        self.heap = []
        self.size = 0
        heapify(self.heap)

    def push(self, elem):
        heappush(self.heap, (self.order * elem[0], elem[1]))
        self.size += 1

    def pop(self):
        pop = heappop(self.heap)
        self.size -= 1
        return pop[0] * self.order, pop[1]

    def pop_item(self):
        return self.pop()[1]

    def is_empty(self):
        return self.size == 0


class Dataset:
    """Utility class to manage a dataset stored in a external file."""
    classes = []
    transactions = []
    nb_transactions = 0

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan"):
        """reads the dataset files and initializes the dataset"""
        try:
            if algo == "PrefixSpan":
                self.pid = []
                self.projection = []
                self.support = [] # list containing positive and negative support
                last_position = sys.maxsize
                tid = -1
                for path_id, filepath in enumerate([filepath_positive, filepath_negative]):
                    path_id = POS if path_id == 0 else NEG

                    lines = [line.strip() for line in open(filepath, "r")]
                    lines = [line.split(" ") for line in lines if line]  # skipping blank lines

                    for line in lines:
                        symbol, position = line
                        position = int(position)
                        if position < last_position:
                            tid += 1
                            self.classes.append(path_id)
                            self.transactions.append(list())
                            self.nb_transactions += 1
                            self.pid.append(-1)
                        self.transactions[tid].append(symbol)
                        last_position = position

        except IOError as e:
            print("Unable to read file !\n" + e)
    
    def __repr__(self):
        return str(self.transactions).replace('(', '\n(')

    def __str__(self):
        return str(self.projection).replace('\'', '') + ' ' + str(self.support[0]) + ' ' + str(self.support[1]) + ' ' + str(sum(self.support))

    def __lt__(self, other):
        return False

    def branch(self, symbol, pos_support, neg_support):
        branch = deepcopy(self) # TODO checkup
        branch.projection.append(symbol)
        branch.support = [pos_support, neg_support]

        # advance pid
        for tid, transaction in enumerate(branch.transactions):
            j = 1
            pid = branch.pid[tid]
            length = len(transaction)
            while pid + j < length:
                if transaction[pid + j] == symbol:
                    break
                else:
                    j += 1
            branch.pid[tid] += j
            if branch.pid[tid] >= length:
                del transaction
                self.nb_transactions -= 1

        return branch

    def compute_support(self):
        support_symbols_pos = defaultdict(int)
        support_symbols_neg = defaultdict(int)
        for tid, transaction in enumerate(self.transactions) :
            for symbol in set(transaction[self.pid[tid]+1:]):
                if self.classes[tid] == POS:
                    support_symbols_pos[symbol] += 1
                elif self.classes[tid] == NEG:
                    support_symbols_neg[symbol] += 1
        
        return support_symbols_pos, support_symbols_neg


def prefixspan(dataset, k):
    queue = Heap(order='max')
    queue.push( (sys.maxsize, dataset) )

    last_support = sys.maxsize
    while not queue.is_empty() and k >= 0:
        support, node = queue.pop()
        if support < last_support:
            last_support = support
            k -= 1
            if k < 0:
                break
        if node.support:
            print(node)
            
        support_symbols_pos, support_symbols_neg = node.compute_support()
        for symbol in set().union(support_symbols_pos, support_symbols_neg):
            queue.push((
                support_symbols_pos[symbol] + support_symbols_neg[symbol],
                node.branch(symbol, support_symbols_pos[symbol], support_symbols_neg[symbol])
            ))


def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])

    # read the dataset files 
    dataset = Dataset(pos_filepath, neg_filepath, algo="PrefixSpan")
    prefixspan(dataset, k)


if __name__ == "__main__":
    main()
