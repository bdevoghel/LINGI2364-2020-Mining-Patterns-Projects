#!/usr/bin/env python3

import sys
from time import time
from copy import copy
from collections import Counter, OrderedDict, defaultdict
from heapq import heappop, heappush, heapify
import math

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
        for item in self.heap:
            if item[0] * self.order == score:
                return True
        return False


class ClosedHeap(Heap):
    """
    Extention for closed patterns in a Heap
    """

    def contains_super_pattern(self, pattern):
        for item in self.heap:
            if is_super_pattern_node(item[1], pattern):
                return True
        return False

    def contains_closed_super_pattern(self, pattern):
        for item in self.heap:
            if is_super_pattern_node(item[1], pattern):
                if item[1].support == pattern.support:
                    return True
        return False

    def contains_closed_under_pattern(self, pattern):
        for item in self.heap:
            if is_super_pattern_node(pattern, item[1]):
                if item[1].support == pattern.support:
                    return True
        return False

    def remove_contained_closed_under_pattern(self, pattern):
        to_remove = []
        for item in self.heap:
            if is_super_pattern_node(pattern, item[1]):
                if item[1].support == pattern.support:
                    # item is an under pattern that has to be removed
                    to_remove.append(item)
        for tr in to_remove:
            self.heap.remove(tr)
            self.size -= 1
        return

def impurity_entropy(x): #information gain
    if x==1 or x == 0:
        return 0
    return -x* math.log(x) - (1-x)*math.log(1-x)

class Dataset:
    """
    Data structure to mine frequent sequences in a dataset stored in external class files (one + and one -).

    Mining DFS-wise. Every instance is a node in the DFS.
    """
    classes = []
    transactions = []
    P = 0
    N = 0
    wracc_factor = None  # computed at end of init() : self.N*self.P/float((self.P+self.N)**2)
    imp_term = None

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan", score_type="supsum"):
        """reads the dataset files and initializes the dataset"""
        try:
            if algo == "PrefixSpan":
                self.score_type = score_type
                self.pid = []
                self.projection = []
                self.support = ()  # 2-tuple containing positive and negative support

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
                            self.classes.append(path_id)
                            self.pid.append(-1)  # init
                        self.transactions[tid].append(symbol)
                        last_position = position

                self.wracc_factor = self.N * self.P / float((self.P + self.N) ** 2)
                self.imp_term = impurity_entropy(self.P/(self.P + self.N))
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
            for symbol in set(transaction[self.pid[tid] + 1:]):
                if self.classes[tid] == POS:
                    support_symbols_pos[symbol] += 1
                elif self.classes[tid] == NEG:
                    support_symbols_neg[symbol] += 1

        return (support_symbols_pos, support_symbols_neg)

    def advance_pid(self, symbol):
        """
        Last step of the PrefixSpan algo.
        Setting the pid to the first occurence (after prev_pid) of the symbol branched on.
        """
        # NOTE takes 75%+ of computing, should be optimize with <ordered list of last position of symbols> or <list of symbol positions>
        # cf slide 17-18 of frequent sequence mining
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
        branch.support = (pos_support, neg_support)
        return branch

    def score(self, pos_support=None, neg_support=None):
        """
        Computes score based on type precised at init().
        If pos_support and neg_support are given, computes score based on these rather than self.support.
        """
        if pos_support is None and neg_support is None:
            if not self.support:
                pos_support = self.P
                neg_support = self.N
            else :
                pos_support = self.support[0]
                neg_support = self.support[1]

        if self.score_type == "supsum":
            return pos_support + neg_support
        if self.score_type == "wracc":
            return round(self.wracc_factor * ((pos_support / self.P) - (neg_support / self.N)), 5)
        if self.score_type == "abswracc":
            return abs(round(self.wracc_factor * ((pos_support / self.P) - (neg_support / self.N)), 5))
        if self.score_type == "infogain":
            if pos_support == self.P and neg_support == self.N: #root and symbol present everywhere
                return 0
            else:
                return self.imp_term\
                   - (pos_support + neg_support)/(self.P + self.N) * impurity_entropy(pos_support/(pos_support + neg_support)) \
                   - (self.P + self.N - pos_support - neg_support)/(self.P + self.N) * impurity_entropy((self.P - pos_support)/(self.P + self.N - pos_support - neg_support))

    def satisfies_heuristic(self, min_p, max_n, min_n, pos_support=None, neg_support=None):
        """
        Returns True if support heuristic is satisfied.
        If pos_support and neg_support are given, computes score based on these rather than self.support.
        """
        if pos_support is None and neg_support is None:
            if not self.support: return True  # root
            pos_support = self.support[0]
            neg_support = self.support[1]

        if self.score_type == "wracc":
            return pos_support >= min_p or neg_support <= max_n
        elif self.score_type == "abswracc":
            return pos_support >= min_p or neg_support >= min_n
        else :
            return True


def add_pattern(heap, nb_different_scores, k, score, node, min_p, max_n, min_n):
    if not heap.contains_closed_super_pattern(node):
        if nb_different_scores < k:
            if not heap.contains(score):  # pattern should be added without decrementing k
                nb_different_scores += 1
            heap.remove_contained_closed_under_pattern(node)
            heap.push((score, node))

        else:  # reached k, only add if score is already present
            min_wracc, __ = heap.peek()
            if heap.contains(score):  # score is already present
                heap.remove_contained_closed_under_pattern(node)
                heap.push((score, node))
            elif score > min_wracc:  # score is more interesting than what already exists
                new_min_wracc, __ = heap.peek()
                while (new_min_wracc == min_wracc):
                    heap.pop()  # removes worse scores
                    new_min_wracc, __ = heap.peek()

                min_p = node.P * new_min_wracc / node.wracc_factor
                max_n = -node.N * new_min_wracc / node.wracc_factor
                min_n = node.N * new_min_wracc / node.wracc_factor
                heap.remove_contained_closed_under_pattern(node)
                heap.push((score, node))
    return nb_different_scores, min_p, max_n, min_n

def is_super_pattern_node(super_pattern, pattern):  # ex ABCD is super pattern of BC
    super_pattern_items = super_pattern.projection
    pattern_items = pattern.projection
    i = 0
    j = 0
    while i < len(pattern_items) and j < len(super_pattern_items):
        if (pattern_items[i] == super_pattern_items[j]):
            i += 1
        j += 1
    return i == len(pattern_items) and super_pattern.score() == pattern.score()  # then we have found for each item in pattern, the same in super_pattern in the same order


def prefixspan(dataset, k):
    queue = Heap(order="max")  # max heap for DFS with heuristic
    queue.push((sys.maxsize, dataset))  # root

    k_best_wracc = ClosedHeap(order="min")  # min heap containting the curently best solutions
    k_best_wracc.push((-sys.maxsize, dataset))  # root
    nb_different_scores = 1  # counts number of different scores there are in k_best_wracc, should be <= k

    # support heuristic values for DFS
    min_p = 0
    max_n = 0
    min_n = 0

    # compute DFS tree with heuristic
    while not queue.is_empty():
        score, node = queue.pop()
        if node.satisfies_heuristic(min_p, max_n, min_n):
            if node.support:  # skip for root
                # add node to k_best_wracc if its score is better than what exists and the closed is not present
                nb_different_scores, min_p, max_n, min_n = add_pattern(k_best_wracc, nb_different_scores, k, score, node,
                                                                min_p, max_n, min_n)

            # DFS branching
            support_symbols_pos, support_symbols_neg = node.compute_support()
            for symbol in (support_symbols_pos + support_symbols_neg):
                pos_supp, neg_supp = support_symbols_pos[symbol], support_symbols_neg[symbol]

                if node.satisfies_heuristic(min_p, max_n, min_n, pos_supp, neg_supp):
                    new_score = node.score(pos_support=pos_supp, neg_support=neg_supp)
                    queue.push((new_score, node.branch(symbol, pos_supp, neg_supp)))

    # print k best patterns
    #print(len(k_best_wracc.heap))
    for item in sorted(k_best_wracc.heap, reverse=True):
        print(item[1])


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])

    dataset = Dataset(pos_filepath, neg_filepath, score_type="infogain")  # choose from : supsum, wrac, abswracc, infogain
    prefixspan(dataset, k)


if __name__ == "__main__":
    # "Datasets/Test/positive.txt" "Datasets/Test/negative.txt" 3
    # "Datasets/Protein/SRC1521.txt" "Datasets/Protein/PKA_group15.txt" 3
    # "Datasets/Reuters/earn.txt" "Datasets/Reuters/acq.txt" 3

    # start_time = time()
    main()
    # print(f"--- {time() - start_time)} seconds ---")
