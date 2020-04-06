#!/usr/bin/env python3

import sys
from copy import deepcopy
from collections import OrderedDict, defaultdict
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

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan"):
        """reads the dataset files and initializes the dataset"""
        try:
            if algo == "PrefixSpan":
                self.algo = algo
                self.transactions = OrderedDict()
                last_position = sys.maxsize
                tid = 0
                for path_id, filepath in enumerate([filepath_positive, filepath_negative]):
                    path_id = POS if path_id == 0 else NEG

                    lines = [line.strip() for line in open(filepath, "r")]
                    lines = [line.split(" ") for line in lines if line]  # skipping blank lines

                    for line in lines:
                        symbol, position = line
                        position = int(position)
                        if position < last_position:
                            tid += 1
                            self.transactions[tid] = {'symbols':[], 'pid':-1, 'class':path_id}
                        self.transactions[tid]['symbols'].append(symbol)
                        last_position = position
                self.projection = []
                self.freq = None # not computed

        except IOError as e:
            print("Unable to read file !\n" + e)
    
    def __repr__(self):
        return str(self.transactions).replace('(', '\n(')

    def __str__(self):
        return str(self.projection).replace('\'', '') + ' ' + str(self.freq['pos']) + ' ' + str(self.freq['neg']) + ' ' + str(sum(self.freq.values()))

    def __cmp__(self, other):
        return len(self.transactions) - len(other.transactions)

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def branch(self, symbol, freq_pos, freq_neg):
        new = deepcopy(self)
        new.projection.append(symbol)
        new.freq = {'pos':freq_pos, 'neg':freq_neg}

        # advance pid
        for transaction in new.transactions.values():
            j = 1
            pid = transaction['pid']
            while pid + j < len(transaction['symbols']):
                if transaction['symbols'][pid + j] == symbol:
                    break
                else:
                    j += 1
            transaction['pid'] += j
            if transaction['pid'] >= len(transaction['symbols']):
                del transaction

        return (freq_pos + freq_neg, new)


def prefixspan(dataset, k):
    queue = Heap(order='max')
    queue.push( (sys.maxsize, dataset) )

    last_freq = sys.maxsize
    while not queue.is_empty() and k >= 0:
        freq, node = queue.pop()
        if freq < last_freq:
            last_freq = freq
            k -= 1
            if k < 0:
                break
        if node.freq is not None:
            print(node)
            

        freq_symbols_pos = defaultdict(int)
        freq_symbols_neg = defaultdict(int)
        for transaction in node.transactions.values() :
            for symbol in set(transaction['symbols'][transaction['pid']+1:]):
                if transaction['class'] == POS:
                    freq_symbols_pos[symbol] += 1
                elif transaction['class'] == NEG:
                    freq_symbols_neg[symbol] += 1

        for symbol in set(freq_symbols_pos.keys()).union(freq_symbols_neg.keys()):
            queue.push(node.branch(symbol, freq_symbols_pos[symbol], freq_symbols_neg[symbol]))


def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])

    # read the dataset files 
    dataset = Dataset(pos_filepath, neg_filepath, algo="PrefixSpan")
    prefixspan(dataset, k)


if __name__ == "__main__":
    main()
