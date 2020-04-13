#!/usr/bin/env python3

import sys
from time import time
from copy import copy
from collections import Counter, OrderedDict, defaultdict, deque
from heapq import heappop, heappush, heapify


"""
Algo based on PrefixSpan with Weighted relative accuracy

Finds the k best paterns that are highly present in the positive class, but not in the negative class.
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


def next_larger_element(a:list, element, default=None):
    for e in a:
        if e > element:
            return e
    return default


class Dataset:
    """
    Data structure for the transactions in a dataset stored in external class files (+ and -).
    """

    def __init__(self, filepath_positive, filepath_negative, score_type="supsum"):
        """reads the dataset files and initializes the dataset"""
        try:
            self.score_type = score_type
            self.transactions = []
            self.nb_transactions = None  # computed at end of init()
            self.transactions_len = []
            self.symbols_positions = []
            self.classes = []
            self.P = 0
            self.N = 0
            self.wracc_factor = None  # computed at end of init()
            
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
                        self.symbols_positions.append(defaultdict(list))
                        self.classes.append(path_id)
                    self.transactions[tid].append(symbol)
                    self.symbols_positions[tid][symbol].append(position-1)
                    last_position = position
            self.nb_transactions = tid
            for tr in self.transactions:
                self.transactions_len.append(len(tr))
            self.wracc_factor = self.N*self.P / float((self.P+self.N)**2)
            self.impurity_factor = impurity_entropy(self.P / (self.P+self.N))
        except IOError as e:
            print("Unable to read file !\n" + e)


class Node:
    """
    Node of a BFS tree for mining frequent sequences in a dataset.
    """

    def __init__(self, dataset:Dataset, sequence=None, pid=None, p=None, n=None):
        self.dataset = dataset
        self.p = dataset.P if p is None else p  # support in positive class
        self.n = dataset.N if n is None else n  # support in negative class
        self.sequence = [] if sequence is None else sequence
        self.pid = [-1] * len(dataset.transactions) if pid is None else pid
        self.is_root = sequence is None

    def __repr__(self):
        return str(self.sequence)

    def __str__(self):
        """formated print of the node's sequence"""
        return str(self.sequence).replace("\'", "") + f" {self.p} {self.n} {self.score()}"

    def __lt__(self, other):
        """has no order"""
        return False

    def compute_support(self):
        """
        First step of the PrefixSpan algo.
        Traverse every transaction from pid+1 and increase counter of encountered symbol.
        """
        symbols_pos_supp = Counter()
        symbols_neg_supp = Counter()
        for tid, transaction in enumerate(self.dataset.transactions):
            for symbol in set(transaction[self.pid[tid]+1:]):
                if self.dataset.classes[tid] == POS:
                    symbols_pos_supp[symbol] += 1
                elif self.dataset.classes[tid] == NEG:
                    symbols_neg_supp[symbol] += 1
        
        return (symbols_pos_supp, symbols_neg_supp)

    def advance_pid(self, symbol):
        """
        Last step of the PrefixSpan algo.
        Setting the pid to the first occurence (after prev_pid) of the symbol branched on.
        Uses a list of symbol positions to faster update the pid than going through the whole transaction.
        """
        new_pid = [None] * (self.dataset.nb_transactions+1)
        for tid, symbols_position in enumerate(self.dataset.symbols_positions):
            if symbol in symbols_position:
                new_pid[tid] = next_larger_element(symbols_position[symbol], self.pid[tid], default=self.dataset.transactions_len[tid])
            else:
                new_pid[tid] = self.dataset.transactions_len[tid]
        return new_pid

    def branch(self, symbol, pos_supp, neg_supp):
        """branches self on symbol, returns new Node instance"""
        return Node(self.dataset, sequence=(self.sequence + [symbol]), pid=self.advance_pid(symbol), p=pos_supp, n=neg_supp)

    def score(self, pos_supp=None, neg_supp=None):
        """
        Computes score based on type precised in dataset. Score rounded at 5 decimals.
        If pos_supp and neg_supp are given, computes score based on these rather than on self.p and self.n.
        """
        if pos_supp is None and neg_supp is None:
            if self.is_root:
                pos_supp = self.dataset.P
                neg_supp = self.dataset.N
            else:
                pos_supp = self.p
                neg_supp = self.n
        
        if self.dataset.score_type == "supsum":
            return pos_supp + neg_supp
        if self.dataset.score_type == "wracc":
            return round(self.dataset.wracc_factor * ( (pos_supp/self.dataset.P) - (neg_supp/self.dataset.N)), 5)
        if self.dataset.score_type == "abswracc":
            return abs(round(self.dataset.wracc_factor * ( (pos_supp/self.dataset.P) - (neg_supp/self.dataset.N)), 5))
        if self.dataset.score_type == "infogain":
            if self.is_root:
                return 0
            else:
                a = pos_supp + neg_supp
                b = self.dataset.P + self.dataset.N
                c = b - pos_supp - neg_supp
                return round(self.dataset.impurity_factor - a/b * impurity_entropy(pos_supp / a) - c/b * impurity_entropy((self.dataset.P-pos_supp) / c), 5)
        else:
            raise NotImplementedError("Unknown scoring type")

    def satisfies_heuristic(self, min_p, max_n, min_n, pos_supp=None, neg_supp=None):
        """
        Returns True if support heuristic is satisfied.
        If pos_supp and neg_supp are given, computes heuristic based on these rather than on self.p and self.n.
        """
        if pos_supp is None and neg_supp is None:
            if self.is_root: return True
            pos_supp = self.p
            neg_supp = self.n
        
        if self.dataset.score_type == "wracc":
            return pos_supp >= min_p or neg_supp <= max_n
        elif self.dataset.score_type == "abswracc":
            return pos_supp >= min_p or neg_supp >= min_n
        else :
            return True


def add_sequence(heap, nb_different_scores, k, score, node, min_p, max_n, min_n):
    if nb_different_scores < k:
        if not heap.contains(score):  # sequence should be added without decrementing k
            nb_different_scores += 1
        heap.push((score, node))

    else:  # reached k sequences in heap, only add if score is already present
        min_wracc, __ = heap.peek()
        if heap.contains(score):  # score is already present
            heap.push((score, node))
        elif score > min_wracc:  # score is more interesting than what already exists
            new_min_wracc, __ = heap.peek()
            while(new_min_wracc == min_wracc):
                heap.pop()  # removes worst score
                new_min_wracc, __ = heap.peek()
            heap.push((score, node))
            
            min_p = node.dataset.P * new_min_wracc / node.dataset.wracc_factor
            min_n = node.dataset.N * new_min_wracc / node.dataset.wracc_factor
            max_n = -min_n
    return nb_different_scores, min_p, max_n, min_n

def prefixspan(dataset, k):
    root = Node(dataset)
    queue = Heap(order="max")  # max heap for BFS node exploration with heuristic
    queue.push((sys.maxsize, root))  # adding root

    k_best_wracc = Heap(order="min")  # min heap containting the currently k best solutions
    k_best_wracc.push((-sys. maxsize, root))  # adding root with worst score
    nb_different_scores = 1  # counts number of different scores there are in k_best_wracc, should always be <= k

    # support heuristic values for BFS
    min_p = 0
    max_n = 0
    min_n = 0

    # compute BFS tree with heuristic
    while not queue.is_empty():
        score, node = queue.pop()
        if node.satisfies_heuristic(min_p, max_n, min_n):
            if not node.is_root:  # skip for root
                # add node to k_best_wracc if its score is better than what exists
                nb_different_scores, min_p, max_n, min_n = add_sequence(k_best_wracc, nb_different_scores, k, score, node, min_p, max_n, min_n)

            # BFS branching
            symbols_pos_supp, symbols_neg_supp = node.compute_support()
            for symbol in OrderedDict(sorted((symbols_pos_supp + symbols_neg_supp).items(), key=lambda item:(item[1], item[0]), reverse=True)):
                pos_supp, neg_supp = symbols_pos_supp[symbol], symbols_neg_supp[symbol]

                if node.satisfies_heuristic(min_p, max_n, min_n, pos_supp, neg_supp):
                    new_score = node.score(pos_supp=pos_supp, neg_supp=neg_supp)
                    queue.push((new_score, node.branch(symbol, pos_supp, neg_supp)))

    # print k best sequences
    for item in sorted(k_best_wracc.heap, reverse=True):
        print(item[1])


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])
    try:
        score_type = sys.argv[4]
    except IndexError:
        score_type = "wracc" # choose from : wracc, abswracc, infogain

    dataset = Dataset(pos_filepath, neg_filepath, score_type=score_type)
    prefixspan(dataset, k)


if __name__ == "__main__":
    main()
