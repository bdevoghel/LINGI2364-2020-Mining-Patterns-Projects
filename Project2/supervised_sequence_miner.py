#!/usr/bin/env python3

import sys
import time
from copy import copy
from collections import Counter, OrderedDict, defaultdict
from heapq import heappop, heappush, heapify


"""
Algo based on PrefixSpan with Weighted relative accuracy  - TODO step to COP

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
        return pop[0] * self.order, pop[1]

    def pop_item(self):
        return self.pop()[1]

    def is_empty(self):
        return self.size == 0

    def peek(self):
        return self.heap[0][0] * self.order, self.heap[0][1]

    def contains(self, wracc):
        Found = False
        i = 0
        while not Found and i < self.size:
            if(self.heap[i][0]  * self.order == wracc):
                Found = True
            i += 1
        return Found



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

    def __init__(self, filepath_positive, filepath_negative, algo="PrefixSpan", score_type="supsum"):
        """reads the dataset files and initializes the dataset"""
        try:
            if algo == "PrefixSpan" or algo == "BranchAndBound":
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
                self.wracc_factor = self.N*self.P/float((self.P+self.N)**2)
        except IOError as e:
            print("Unable to read file !\n" + e)
    
    def __repr__(self):
        return str(self.projection) # or str(self.transactions).replace("(", "\n(")

    def __str__(self):
        """formated print of the node"s frequent sequence"""
        return str(self.projection).replace("\'", "") + " " + str(self.support[0]) + " " + str(self.support[1]) + " " + str(self.score())

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
        #MOYEN D'AMELIORER CFR sl 17-18 de frequent sequence mining
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
        """compute score based on type precised at init(), can be done on other support than self"""
        if pos_support is None and neg_support is None:
            pos_support = self.support[0]
            neg_support = self.support[1]
        
        if self.score_type == "supsum":
            return pos_support + neg_support
        if self.score_type == "wracc":
            #return float("{:.5f}".format(self.wracc_factor * ( (pos_support/self.P) - (neg_support/self.N) )))
            return self.wracc_factor * ( (pos_support/self.P) - (neg_support/self.N) )


def prefixspanOLD(dataset, k, support_threshold="anti_monotonic"):
    queue = Heap(order="max")  # for DFS with heuristic
    queue.push( (sys.maxsize, dataset) )  # root

    solutions = Heap(order="max")

    last_score = sys.maxsize
    while not queue.is_empty() and k >= 0:
        score, node = queue.pop()
        if support_threshold == "anti_monotonic":
            if score < last_score:
                last_score = score
                k -= 1
                if k < 0:
                    break
            if node.support:  # skip for root
                print(node)
        elif node.support:
            solutions.push((score, str(node)))  # str to save space
            
        support_symbols_pos, support_symbols_neg = node.compute_support()
        all_support = OrderedDict(sorted((Counter(support_symbols_pos) + Counter(support_symbols_neg)).items(), key=lambda item: item[1], reverse=True))  # merge and sort support

        for i, symbol in enumerate(all_support):
            supp_pos, supp_neg = support_symbols_pos[symbol], support_symbols_neg[symbol]
            new_score = node.score(pos_support=supp_pos, neg_support=supp_neg)
            if support_threshold == "anti_monotonic" and (k - i < 0 or (k < queue.size and new_score < queue.heap[k][0])):  # heuristic, only if support threshold is anti-monotonic
                break
            queue.push((new_score, node.branch(symbol, supp_pos, supp_neg)))

    while support_threshold != "anti_monotonic" and k >= 0:
        score, seq = solutions.pop()
        if score < last_score:
            last_score = score
            k -= 1
            if k < 0:
                break
        print(seq)



def prefixspan(dataset, k, support_threshold="anti_monotonic"):
    queue = Heap(order="max")  # for DFS with heuristic
    queue.push((sys.maxsize, dataset))  # root
    #min heap qui va contenir les k meilleur Wracc, ainsi on va calculer les min_p et min_n pour etre au mieux que celui-la
    # on modifie le moins bon des k, quand on trouve mieux

    k_biggest_wracc = Heap(order="min")
    k_biggest_wracc.push((-sys. maxsize, dataset))  # root
    differentScores = 1 #va calculer le nombre de scores differents qu'on a, il doit etre <= k

    min_p = 0
    max_n = 0

    while not queue.is_empty():
        score, node = queue.pop()
        if node.support:  # skip for root
            if differentScores < k: #faut etre sur d'en avoir k diff!!
                if not k_biggest_wracc.contains(score): # we found same W_racc
                    differentScores += 1
                k_biggest_wracc.push((score, node))

            else: # size = k
                min_Wracc, nodeWracc = k_biggest_wracc.peek()
                if k_biggest_wracc.contains(score): # we found same W_racc
                    k_biggest_wracc.push((score, node))  # just add new one

                elif score > min_Wracc: # we found a better W_racc
                    new_min_Wracc, new_nodeWracc = k_biggest_wracc.peek()
                    while(new_min_Wracc == min_Wracc):
                        k_biggest_wracc.pop() #goodbye old oneS with same score
                        new_min_Wracc, new_nodeWracc = k_biggest_wracc.peek()
                    min_p = (score/dataset.wracc_factor)*dataset.P
                    max_n = -((score/dataset.wracc_factor)*dataset.N)
                    k_biggest_wracc.push((score, node))

        support_symbols_pos, support_symbols_neg = node.compute_support()
        #all_support = (Counter(support_symbols_pos) + Counter(support_symbols_neg)).items() #merge

        for symbol in support_symbols_pos: #A MODIF
            supp_pos, supp_neg = support_symbols_pos[symbol], support_symbols_neg[symbol]
            new_score = node.score(pos_support=supp_pos, neg_support=supp_neg)
            if supp_pos >= min_p or supp_neg <= max_n:
                queue.push((new_score, node.branch(symbol, supp_pos, supp_neg)))

    """last_score = sys.maxsize
    while k >= 0:
        score, seq = k_biggest_wracc.pop() #attention c'est pas les plus grand c'est les plus petits
        if score < last_score:
            last_score = score
            k -= 1
            if k < 0:
                break
        print(seq)
    """
    while(not k_biggest_wracc.is_empty()):
        score, seq = k_biggest_wracc.pop()
        print(seq)


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])

    dataset = Dataset(pos_filepath, neg_filepath, algo="BranchAndBound", score_type="wracc")
    prefixspan(dataset, k, support_threshold="non_monotonic")


if __name__ == "__main__":
    #"Datasets/Test/positive.txt" "Datasets/Test/negative.txt" 3
    #"Datasets/Protein/SRC1521.txt" "Datasets/Protein/PKA_group15.txt" 3

    #start_time = time.time()
    main()
    #print("--- %s seconds ---" % (time.time() - start_time))
