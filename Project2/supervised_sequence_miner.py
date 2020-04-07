#!/usr/bin/env python3

import sys
from copy import copy
from collections import Counter, OrderedDict, defaultdict
from heapq import heappop, heappush, heapify

"""
Algo based on Weighted relative accuracy

Finds the k best paterns that are highly present in the positive class, but not in the negative class.

=> consider Wracc score
"""

POS = 1
NEG = -1

class Heap:
    """
    Data structure easing the use of a min- or max-heap.
    Stored items are 2-tuples with the first element being the order.
    """

    def __init__(self, order='min'):
        self.order = -1 if order == 'max' else 1
        self.heap = []
        self.size = 0
        heapify(self.heap)

    def push(self, item):
        heappush(self.heap, (self.order * item[0], item[1]))
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
    """
    Data structure to mine frequent sequences in a dataset stored in external class files (one + and one -).
    
    Mining DFS-wise. Every instance is a node in the DFS.
    """
    classes = []
    transactions = []

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan"):
        """reads the dataset files and initializes the dataset"""
        try:
            if algo == "PrefixSpan":
                self.pid = []
                self.projection = []
                self.support = () # 2-tuple containing positive and negative support

                last_position = sys.maxsize
                tid = -1
                for path_id, filepath in enumerate([filepath_positive, filepath_negative]):
                    path_id = POS if path_id == 0 else NEG  # positive or negative class

                    # reading the file, skipping blank lines
                    lines = [line.strip() for line in open(filepath, "r")]
                    lines = [line.split(" ") for line in lines if line]

                    for line in lines:
                        symbol, position = line  # every line is a <symbol> <position in transaction> pair
                        position = int(position)
                        if position < last_position:  # new transaction beginning
                            tid += 1
                            self.transactions.append(list())
                            self.classes.append(path_id)
                            self.pid.append(-1)  # init
                        self.transactions[tid].append(symbol)
                        last_position = position
        except IOError as e:
            print("Unable to read file !\n" + e)
    
    def __repr__(self):
        return str(self.transactions).replace('(', '\n(')

    def __str__(self):
        """formated print of the node's frequent sequence"""
        return str(self.projection).replace('\'', '') + ' ' + str(self.support[0]) + ' ' + str(self.support[1]) + ' ' + str(sum(self.support))

    def __lt__(self, other):
        return False

    def compute_support(self):
        """
        First step of the PrefixSpan algo.
        Traverse every transaction from pid+1 and increase counter of encountered symbol.
        """
        support_symbols_pos = defaultdict(int)
        support_symbols_neg = defaultdict(int)
        for tid, transaction in enumerate(self.transactions) :
            for symbol in set(transaction[self.pid[tid]+1:]):
                if self.classes[tid] == POS:
                    support_symbols_pos[symbol] += 1
                elif self.classes[tid] == NEG:
                    support_symbols_neg[symbol] += 1
        
        return support_symbols_pos, support_symbols_neg

    def advance_pid(self, symbol): 
        """
        Last step of the PrefixSpan algo.
        Setting the pid to the first occurence (after prev_pid) of the symbol branched on.
        """
        # NOTE takes 75%+ of computing, should be optimize with <ordered list of last position of symbols> or <list of symbol positions>
        new_pid = [None] * len(self.pid)
        for tid, transaction in enumerate(self.transactions):
            j = 1
            pid = self.pid[tid]
            length = len(transaction)
            while pid + j < length:
                if transaction[pid + j] == symbol:
                    break
                else:
                    j += 1
            new_pid[tid] = pid + j
        return new_pid

    def branch(self, symbol, pos_support, neg_support):
        """branches self on symbol, returns new class instance"""
        branch = copy(self)
        branch.pid = self.advance_pid(symbol)
        branch.projection = self.projection + [symbol]
        branch.support = [pos_support, neg_support]
        return branch


def prefixspan(dataset, k):
    queue = Heap(order='max')  # for DFS with heuristic
    queue.push( (sys.maxsize, dataset) )  # root

    last_support = sys.maxsize
    while not queue.is_empty() and k >= 0:
        support, node = queue.pop()
        if support < last_support:
            last_support = support
            k -= 1
            if k < 0:
                break
        if node.support:  # skip for root
            print(node)
            
        support_symbols_pos, support_symbols_neg = node.compute_support()
        all_support = OrderedDict(sorted((Counter(support_symbols_pos) + Counter(support_symbols_neg)).items(), key=lambda item: item[1], reverse=True))  # merge and sort support

        for i, symbol in enumerate(all_support):
            supp_pos, supp_neg = support_symbols_pos[symbol], support_symbols_neg[symbol]
            sum_supp = supp_pos + supp_neg
            if k - i < 0 or (k < queue.size and sum_supp < queue.heap[k][0]): break # heuristic
            queue.push((sum_supp, node.branch(symbol, supp_pos, supp_neg)))

def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])

    dataset = Dataset(pos_filepath, neg_filepath, algo="PrefixSpan")
    prefixspan(dataset, k)


if __name__ == "__main__":
    main()
