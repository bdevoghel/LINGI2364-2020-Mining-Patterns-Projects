#!/usr/bin/env python3

import sys
from time import time
from copy import copy
from collections import Counter, OrderedDict, defaultdict, deque
from heapq import heappop, heappush, heapify


"""
Algo based on PrefixSpan with Weighted relative accuracy

Finds the k best paterns that are highly present in the positive class, but not in the negative class.

=> consider Wracc score
"""

POS = 1
NEG = -1

class Heap:
    """
    Data structure easing the use of a min- or max-heap.
    Stored items are 2-tuples with the first element being the ordering score.
    """

    def __init__(self, order="min"):
        self.order = -1 if order == "max" else 1
        self.heap = []
        self.size = 0
        heapify(self.heap)

    def push(self, item):
        heappush(self.heap, (self.order * item[0], item[1]))
        self.size += 1

    def pop(self):
        pop = heappop(self.heap)
        self.size -= 1
        return (pop[0] * self.order, pop[1])

    def pop_item(self):
        return self.pop()[1]

    def is_empty(self):
        return self.size == 0

    def peek(self):
        if not self.is_empty():
            return (self.heap[0][0] * self.order, self.heap[0][1])

    def contains(self, score):
        for i in range(self.size):
            if self.heap[i][0] * self.order == score:
                return True
        return False



class Dataset:
    """
    Data structure to mine frequent sequences in a dataset stored in external class files (one + and one -).
    
    Mining DFS-wise. Every instance is a node in the DFS.
    """
    classes = []
    transactions = []
    symbols_positions = []
    P = 0
    N = 0
    wracc_factor = None  # computed at end of init() : self.N*self.P/float((self.P+self.N)**2)

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan", score_type="supsum"):
        """reads the dataset files and initializes the dataset"""
        try:
            if algo == "PrefixSpan":
                self.score_type = score_type
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

                    # parsing transactions
                    for line in lines:
                        symbol, position = line  # every line is a <symbol> <position in transaction> pair
                        position = int(position)
                        if position < last_position:  # new transaction beginning
                            tid += 1
                            if path_id == POS:
                                self.P += 1
                            elif path_id == NEG:
                                self.N += 1
                            self.transactions.append(list())
                            self.symbols_positions.append(defaultdict(deque))
                            self.classes.append(path_id)
                            self.pid.append(-1)  # init
                        self.transactions[tid].append(symbol)
                        self.symbols_positions[tid][symbol].append(position-1)
                        last_position = position
                
                self.wracc_factor = self.N*self.P/float((self.P+self.N)**2)
        except IOError as e:
            print("Unable to read file !\n" + e)
    
    def __repr__(self):
        return str(self.projection)

    def __str__(self):
        """formated print of the node"s frequent sequence"""
        return str(self.projection).replace("\'", "") + f" {self.support[0]} {self.support[1]} {self.score()}"

    def __lt__(self, other):
        """has no order"""
        return False

    def compute_support(self):
        """
        First step of the PrefixSpan algo.
        Traverse every transaction from pid+1 and increase counter of encountered symbol.
        """
        support_symbols_pos = Counter()
        support_symbols_neg = Counter()
        for tid, transaction in enumerate(self.transactions):
            for symbol in set(transaction[self.pid[tid]+1:]):
                if self.classes[tid] == POS:
                    support_symbols_pos[symbol] += 1
                elif self.classes[tid] == NEG:
                    support_symbols_neg[symbol] += 1
        
        return (support_symbols_pos, support_symbols_neg)

    def advance_pid(self, symbol): 
        """
        Last step of the PrefixSpan algo.
        Setting the pid to the first occurence (after prev_pid) of the symbol branched on.
        Uses a list of symbol positions to faster update the pid than going through the whole transaction.
        """
        new_pid = [None] * len(self.pid)
        for tid, symbols_position in enumerate(self.symbols_positions):
            if symbols_position[symbol]:
                next_pid = symbols_position[symbol].popleft()
                while next_pid < self.pid[tid]:
                    if not symbols_position[symbol]:
                        next_pid = len(self.transactions[tid])
                        break
                    next_pid = symbols_position[symbol].popleft()
            else:
                next_pid = len(self.transactions[tid])
            new_pid[tid] = next_pid
        return new_pid

    def branch(self, symbol, pos_support, neg_support):
        """branches self on symbol, returns new class instance"""
        branch = copy(self)
        branch.pid = self.advance_pid(symbol)
        branch.projection = self.projection + [symbol]
        branch.support = (pos_support, neg_support)
        return branch

    def score(self, pos_support=None, neg_support=None):
        """
        Computes score based on type precised at init().
        If pos_support and neg_support are given, computes score based on these rather than self.support.
        """
        if pos_support is None and neg_support is None:
            pos_support = self.support[0]
            neg_support = self.support[1]
        
        if self.score_type == "supsum":
            return pos_support + neg_support
        if self.score_type == "wracc":
            return round(self.wracc_factor * ( (pos_support/self.P) - (neg_support/self.N)), 5)

    def satisfies_heuristic(self, min_p, max_n, pos_support=None, neg_support=None):
        """
        Returns True if support heuristic is satisfied.
        If pos_support and neg_support are given, computes score based on these rather than self.support.
        """
        if pos_support is None and neg_support is None:
            if not self.support: return True  # root
            pos_support = self.support[0]
            neg_support = self.support[1]
        
        return pos_support >= min_p or neg_support <= max_n


def add_pattern(heap, nb_different_scores, k, score, node, min_p, max_n):
    if nb_different_scores < k:
        if not heap.contains(score):  # pattern should be added without decrementing k
            nb_different_scores += 1
        heap.push((score, node))

    else: # reached k, only add if score is already present
        min_wracc, __ = heap.peek()
        if heap.contains(score):  # score is already present
            heap.push((score, node))
        elif score > min_wracc:  # score is more interesting than what already exists
            new_min_wracc, __ = heap.peek()
            while(new_min_wracc == min_wracc):
                heap.pop()  # removes worse scores
                new_min_wracc, __ = heap.peek()
            
            min_p = node.P * new_min_wracc / node.wracc_factor
            max_n = -node.N * new_min_wracc / node.wracc_factor
            heap.push((score, node))
    return nb_different_scores, min_p, max_n

def prefixspan(dataset, k):
    queue = Heap(order="max")  # max heap for DFS with heuristic
    queue.push((sys.maxsize, dataset))  # root

    k_best_wracc = Heap(order="min")  # min heap containting the curently best solutions
    k_best_wracc.push((-sys. maxsize, dataset))  # root
    nb_different_scores = 1  # counts number of different scores there are in k_best_wracc, should be <= k

    # support heuristic values for DFS
    min_p = 0
    max_n = 0

    # compute DFS tree with heuristic
    while not queue.is_empty():
        score, node = queue.pop()
        if node.satisfies_heuristic(min_p, max_n):
            if node.support:  # skip for root
                # add node to k_best_wracc if its score is better than what exists
                nb_different_scores, min_p, max_n = add_pattern(k_best_wracc, nb_different_scores, k, score, node, min_p, max_n)

            # DFS branching
            support_symbols_pos, support_symbols_neg = node.compute_support()
            for i, symbol in enumerate(support_symbols_pos + support_symbols_neg):
                pos_supp, neg_supp = support_symbols_pos[symbol], support_symbols_neg[symbol]
                
                if node.satisfies_heuristic(min_p, max_n, pos_supp, neg_supp):
                    new_score = node.score(pos_support=pos_supp, neg_support=neg_supp)
                    queue.push((new_score, node.branch(symbol, pos_supp, neg_supp)))
                    print(i)

    # print k best patterns
    for item in sorted(k_best_wracc.heap, reverse=True):
        print(item[1])


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])

    dataset = Dataset(pos_filepath, neg_filepath, score_type="wracc")
    prefixspan(dataset, k)


if __name__ == "__main__":
    # "Datasets/Test/positive.txt" "Datasets/Test/negative.txt" 3
    # "Datasets/Protein/SRC1521.txt" "Datasets/Protein/PKA_group15.txt" 3
    # "Datasets/Reuters/earn.txt" "/Datasets/Reuters/acq.txt" 3

    # start_time = time()
    main()
    # print(f"--- {time() - start_time)} seconds ---")
