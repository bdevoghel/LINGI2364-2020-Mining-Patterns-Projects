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

    def __init__(self, minsup, database, subsets):
        """
        Initialize the task.
        :param minsup: the minimum positive and negative support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        self.patterns = []  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        self.patterns.append((dfs_code, gid_subsets))

    # Prunes any pattern that is not frequent in the positive and negative class
    def prune(self, gid_subsets):
        # first subset is the set of positive and negative ids
        return len(gid_subsets[0]) + len(gid_subsets[1]) < self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]


def finding_subgraphs(database_file_name_pos, database_file_name_neg, k, minsup):
    """
    Runs gSpan with the specified positive and negative graphs, finds all frequent subgraphs in the positive class
    with a minimum positive support of minsup and prints them.
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
    task = FrequentPositiveAndNegativeGraphs(minsup, graph_database, subsets)  # Creating task

    gSpan(task).run()  # Running gSpan

    # Printing frequent patterns along with their positive support:
	# TODO print k most confident
    for pattern, gid_subsets in task.patterns:
        pos_support = len(gid_subsets[0])  # This will have to be replaced by the confidence and support on both classes
        print('{} {}'.format(pattern, pos_support))


if __name__ == '__main__':

    args = sys.argv
    
    # first parameter: path to positive class file
    # second parameter: path to negative class file
    # third parameter: size of topk
    # fourth parameter: minimum support
    finding_subgraphs(args[1], args[2], k=int(args[3]), minsup=int(args[4]))
	# test with .\Datasets\molecules-small.pos .\Datasets\molecules-small.pos 5 10
