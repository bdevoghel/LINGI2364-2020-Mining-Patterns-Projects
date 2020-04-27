import os
import sys

from sklearn import tree
from sklearn import metrics
import numpy as np
import matplotlib as mpl

from gspan_mining import gSpan
from gspan_mining import GraphDatabase

from collections import Counter, OrderedDict, defaultdict
from heapq import heappop, heappush, heapify
from math import log2


class Heap:
    """
    Data structure easing the use of a min- or max-heap.
    Stored items are 4-tuples with the first 2 elements being 
    the ordering confidence and support and the last element being the item.
    """

    def __init__(self, order="min"):
        self.order = -1 if order == "max" else 1
        self.heap = []
        self.size = 0
        heapify(self.heap)

    def push(self, item):
        heappush(self.heap, (self.order * item[0], self.order * item[1], item[2], item[3]))
        self.size += 1

    def pop(self):
        pop = heappop(self.heap)
        self.size -= 1
        return (pop[0] * self.order, pop[1] * self.order, pop[2], pop[3])

    def pop_item(self):
        pop = self.pop()
        return pop[2], pop[3]

    def is_empty(self):
        return self.size == 0

    def peek(self):
        if not self.is_empty():
            return self.heap[0][0] * self.order, self.heap[0][1] * self.order, self.heap[0][2], self.heap[0][3]

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
        return [(item[0] * self.order, item[1] * self.order, item[2], item[3]) for item in sorted(self.heap, reverse=reverse)]


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
            self.most_confident.push( (confidence, support, dfs_code, gid_subsets) )

        else:  # reached k different sequences in heap, only add if confidence is already present or better

            if self.most_confident.contains_confidence_and_support(confidence, support):  # same confidence and support are already present : add dfs_code
                self.most_confident.push( (confidence, support, dfs_code, gid_subsets) )                
            else:
                min_conf, min_support, __, __ = self.most_confident.peek()
                if confidence > min_conf or (confidence == min_conf and support > min_support):  # dfs_code is more interesting than what already exists : replace worst dfs_code
                    new_min_conf, new_min_support, __, __ = self.most_confident.peek()
                    while new_min_conf == min_conf and new_min_support == min_support:
                        self.most_confident.pop()  # removes worst confidence
                        if(self.most_confident.is_empty()): #Attention, si la queue est devenue vide on peut plus rien pop
                            break
                        new_min_conf, new_min_support, __, __ = self.most_confident.peek()
                    self.min_confidence = new_min_conf
                    self.most_confident.push( (confidence, support, dfs_code, gid_subsets) )

                else: # worse confidence : do not add dfs_code
                    pass

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
        for confidence, support, dfs_code, gid_subsets in self.most_confident.get_all_sorted(reverse=True):
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [np.array(matrix).transpose() for matrix in matrices]


def remove_patternsNOTPROTECTED(database: GraphDatabase, gid_subset):
    newDatabase = database
    for i in gid_subset[0]: #retirer des positives graphs ATTENTION ACCES A PROTECTED VARIABLES
        del newDatabase._graphs[i]
        newDatabase._graph_cnt -= 1
    for i in gid_subset[1]: #retirer des negatives graphs ATTENTION ACCES A PROTECTED VARIABLES
        del newDatabase._graphs[i]
        newDatabase._graph_cnt -= 1
    newDatabase._graphs = {int(i): v for i, v in enumerate(newDatabase._graphs.values())}
    return newDatabase


def train_and_evaluate(nb_rules, minsup, database, subsets):

    list_subsets = []
    for subset in subsets:
        list_subsets.append(list(subset))
    highest_scoring_patterns_rule = []
    rules_gid = []
    for i in range(nb_rules):
        #current_subsets[0] = [t for t in current_subsets[0] if t not in gid_subsets[0]] --> PAS bete! de Charles
        new_subsets = []
        for subset in list_subsets:
            new_subset = []
            for gid in subset:
                print(gid, rules_gid)
                if gid not in rules_gid:
                    new_subset.append(gid)
            new_subsets.append(new_subset)
        task = FrequentPositiveAndNegativeGraphs(minsup, database, new_subsets, k=1)  # Creating task
        gSpan(task).run()  # Running gSpan
        #TODO If several patterns with the same confidence and frequency are found, you should take the lowest in the lexicographical order
        pattern_hold_code, rule_gid = task.most_confident.pop_item()
        highest_scoring_patterns_rule.append(pattern_hold_code)
        rules_gid.extend(rule_gid[0])
        rules_gid.extend(rule_gid[1])

    # Creating feature matrices for training and testing:
    features = task.get_feature_matrices()
    train_fm = np.concatenate((features[0], features[1]))  # Training feature matrix
    train_labels = np.concatenate((np.full(len(features[0]), 1, dtype=int), np.full(len(features[1]), -1, dtype=int)))  # Training labels
    test_fm = np.concatenate((features[2], features[3]))  # Testing feature matrix
    test_labels = np.concatenate((np.full(len(features[2]), 1, dtype=int), np.full(len(features[3]), -1, dtype=int)))  # Testing labels

    classifier = tree.DecisionTreeClassifier(random_state=1)  # Creating model object
    classifier.fit(train_fm, train_labels)  # Training model

    predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

    accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

    # Printing rule patterns:
    # TODO
    # for item in task.most_confident.get_all_sorted(reverse=True):
    #     print_output(item[0], item[1], item[2])

    # Printing classification results:
    print(predicted.tolist())
    print('accuracy: {}'.format(accuracy))
    print()  # Blank line to indicate end of fold.


def print_output(conf, support, dfs_code):
    print(dfs_code, conf, support)


def train_model(database_file_name_pos, database_file_name_neg, nb_rules, minsup, nfolds):
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
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

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            neg_ids,  # Negative training set
            pos_ids,  # Positive test set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate(nb_rules, minsup, graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                np.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                np.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            train_and_evaluate(nb_rules, minsup, graph_database, subsets)


if __name__ == '__main__':

    args = sys.argv
    
    # first parameter: path to positive class file
    # second parameter: path to negative class file
    # third parameter: number of rules for sequential covering
    # fourth parameter: minimum support
    # fifth parameter : number of folds to use in the k-fold cross-validation.
    train_model(args[1], args[2], nb_rules=int(args[3]), minsup=int(args[4]), nfolds=int(args[5]))
