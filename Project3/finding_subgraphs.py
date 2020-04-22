"""
Implementation based on template given [here](https://forge.uclouvain.be/thomascha/LINGI2364-Project3_template/),
and adapted by Mersch-Mersch Severine, and de Voghel Brieuc
"""

"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import pytho

import os
import sys
import numpy
from sklearn import naive_bayes
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase

from collections import Counter, OrderedDict, defaultdict
from heapq import heappop, heappush, heapify
from math import log2


class Heap:
    """
    Data structure easing the use of a min- or max-heap.
    Stored items are 3-tuples with the first 2 elements being 
    the ordering confidence and support and the last element being the item.
    """

    def __init__(self, order="min"):
        self.order = -1 if order == "max" else 1
        self.heap = []
        self.size = 0
        heapify(self.heap)

    def push(self, item):
        heappush(self.heap, (self.order * item[0], self.order * item[1], item[2]))
        self.size += 1

    def pop(self):
        pop = heappop(self.heap)
        self.size -= 1
        return (pop[0] * self.order, pop[1] * self.order, pop[2])

    def pop_item(self):
        return self.pop()[2]

    def is_empty(self):
        return self.size == 0

    def peek(self):
        if not self.is_empty():
            return (self.heap[0][0] * self.order, self.heap[0][1] * self.order, self.heap[0][2])

    def contains_confidence(self, confidence):
        for item in self.heap:
            if item[0] * self.order == confidence:
                return True
        return False

    def contains_confidence_and_support(self, confidence, support):
        for item in self.heap:
            if item[0] * self.order == confidence and item[1] * self.order == support:
                return True
        return False

    def get_all_sorted(self, reverse=False):
        return [(item[0] * self.order, item[1] * self.order, item[2]) for item in sorted(self.heap, reverse=reverse)]


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []
        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")


class FrequentPositiveAndNegativeGraphs(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive and negative graphs.
    This class provides a method to build a feature matrix for each subset.
    """

    def __init__(self, minsup, database, subsets, k):
        """
        Initialize the task.
        :param minsup: the minimum positive and negative support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)

        # The most confident patterns found in the end (as dfs codes represented by strings) with their confidence and support
        self.most_confident = Heap(order="min")

        self.min_support = minsup
        self.min_confidence = 0
        self.gid_subsets = subsets
        self.k = k

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        self.add_sequence(dfs_code, gid_subsets, self.confidence(gid_subsets))

    # Prunes any pattern that is not frequent in the positive and negative class
    def prune(self, gid_subsets):
        # first subset is the set of positive and negative ids
        return not self.satisfies_minsup(gid_subsets)  #or  self.confidence(gid_subsets) < self.min_confidence # TODO CHECK anti-monotonicity

    def confidence(self, gid_subsets):  # not anti-monotonic
        pos_support = len(gid_subsets[0])
        neg_support = len(gid_subsets[1])
        confidence = pos_support / (pos_support + neg_support)
        return confidence

    def satisfies_minsup(self, gid_subsets):  # anti-monotonic
        pos_support = len(gid_subsets[0])
        neg_support = len(gid_subsets[1])
        return pos_support + neg_support >= self.min_support

    def add_sequence(self, dfs_code, gid_subsets, confidence):
        support = len(gid_subsets[0]) + len(gid_subsets[1])
        if self.k > 0:
            if not self.most_confident.contains_confidence_and_support(confidence, support):  # if same confidence and support : sequence should be added without decrementing k
                self.k -= 1
            self.most_confident.push( (confidence, support, dfs_code) )

        else:  # reached k different sequences in heap, only add if confidence is already present or better

            if self.most_confident.contains_confidence_and_support(confidence, support):  # same confidence and support are already present : add dfs_code
                self.most_confident.push( (confidence, support, dfs_code) )                
            else:
                min_conf, min_support, __ = self.most_confident.peek()
                if confidence > min_conf or (confidence == min_conf and support > min_support):  # dfs_code is more interesting than what already exists : replace worst dfs_code
                    new_min_conf, new_min_support, __ = self.most_confident.peek()
                    while new_min_conf == min_conf and new_min_support == min_support:
                        self.most_confident.pop()  # removes worst confidence
                        new_min_conf, new_min_support, __ = self.most_confident.peek()
                    self.min_confidence = new_min_conf
                    self.most_confident.push( (confidence, support, dfs_code) )

                else: # worse confidence : do not add dfs_code
                    pass


def print_output(conf, support, dfs_code):
    print(dfs_code, conf, support)


def finding_subgraphs(database_file_name_pos, database_file_name_neg, k, minsup):
    """
    Runs gSpan with the specified positive and negative graphs, finds k most confident frequent subgraphs in the
    positive class with a minimum support (positive and negative) of minsup and prints them.
    """

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    task = FrequentPositiveAndNegativeGraphs(minsup, graph_database, subsets, k)  # Creating task

    gSpan(task).run()  # Running gSpan

    # Printing k most confident patterns along with their confidence and total support:
    for item in task.most_confident.get_all_sorted(reverse=True):
        print_output(item[0], item[1], item[2])


if __name__ == '__main__':

    args = sys.argv
    
    # first parameter: path to positive class file
    # second parameter: path to negative class file
    # third parameter: size of topk
    # fourth parameter: minimum support
    finding_subgraphs(args[1], args[2], k=int(args[3]), minsup=int(args[4]))
