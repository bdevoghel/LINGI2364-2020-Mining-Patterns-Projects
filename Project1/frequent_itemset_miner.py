"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "Group 10 - MERSCH-MERSCH Severine & DE VOGHEL Brieuc"
"""
from copy import copy, deepcopy
from collections import OrderedDict

class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath, vertical=False, projection=None):
		"""reads the dataset file and initializes files"""

		if projection is not None:
			# Not important
			self.transactions = None
			self.items = None

			initial, item, min_support = projection # original dataset and item on which projection is done

			self.projection = copy(initial.projection)
			self.projection.append(item)

			self.tid_list = OrderedDict()
			for i, tid in enumerate(initial.tid_list.items()):
				if tid[0] <= item: 
					# print('continue')
					continue

				projected_transactions = initial.tid_list[item].intersection(tid[1])
				if len(projected_transactions) >= min_support:
					self.tid_list[tid[0]] = projected_transactions

		else :

			self.transactions = list()
			self.items = {}

			if vertical:
				self.projection = []
				self.tid_list = {} # transaction identifiers

			try:
				lines = [line.strip() for line in open(filepath, "r")]
				lines = [line for line in lines if line]  # Skipping blank lines
				for i, line in enumerate(lines):
					transaction = list(map(int, line.split(" ")))
					self.transactions.append(transaction)
					for item in transaction:
						if (item,) not in self.items:
							self.items[ (item,) ] = 1
							if vertical:
								self.tid_list[item] = {i}
						else:
							self.items[(item,)] +=1
							if vertical:
								self.tid_list[item].add(i)

				if vertical:
					self.tid_list = OrderedDict(sorted(self.tid_list.items(), key=lambda tid: tid[0]))
				
			except IOError as e:
				print("Unable to read dataset file!\n" + e)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return len(self.transactions)

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]

	def get_first_level(self):
		return self.items

	def get_items(self):
		return self.items.keys()

	def project(self, item, min_support):
		projection = Dataset("NO PATH", projection=(self, item, min_support))
		return projection


def transaction_includes_itemset(transaction, itemset): # TODO optimise (is very often called)
	for item in itemset:
		if item not in transaction:
			return False
	return True

def print_frequent_itemsets_from_level(level, nb_transactions):
	for itemset in level.keys():
		if level[itemset] is not None:
			print(str(list(itemset)) + ' (' + str(level[itemset]/nb_transactions) + ')')

def combine(itemset1, itemset2):
	for elem in itemset2:
		if elem not in itemset1:
			itemset1 += (elem,)
	return sorted(itemset1)

def combine_frequent_itemsets(level):
	for i, itemset1 in enumerate(level.keys()):
		if level[itemset1] is None: continue
		for j, itemset2 in enumerate(level.keys()):
			if i >= j or level[itemset2] is None: continue
			new_itemset = combine(itemset1, itemset2)
			if len(new_itemset) > len(itemset1) + 1: continue # TODO should break (only if itemsets are sorted)
			yield new_itemset


def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	dataset = Dataset(filepath)
	min_support = minFrequency * dataset.trans_num()

	prev_level = dataset.get_first_level()
	first_level = True

	while prev_level:

		# COMPUTE SUPPORT OF ITEMSETS IN PREV_LEVEL
		if not first_level:
			for transaction in dataset.transactions:
				for itemset in prev_level.keys():
					if transaction_includes_itemset(transaction, itemset):
						prev_level[itemset] += 1
		else:
			first_level = False
		
		# PRINT FREQUENT ITEMSETS
		for itemset in prev_level.keys():
			if prev_level[itemset] < min_support:
				prev_level[itemset] = None
		print_frequent_itemsets_from_level(prev_level, dataset.trans_num())

		# GENERATE CANDIDATES FOR NEXT LEVEL
		next_level = dict.fromkeys({tuple(x) for x in combine_frequent_itemsets(prev_level)}, 0)
		prev_level = next_level


def eclat(dataset, min_support, nb_transactions):
	for tid in dataset.tid_list.items():
		if len(tid[1]) >= min_support:
			print_frequent_itemset_from_tid(tid, dataset.projection, nb_transactions)
			 
			projected_dataset = dataset.project(tid[0], min_support)
			eclat(projected_dataset, min_support, nb_transactions)

def print_frequent_itemset_from_tid(tid, projection, nb_transactions):
	itemset = copy(projection)
	itemset.append(tid[0])
	print(str(itemset) + ' (' + str(len(tid[1])/nb_transactions) + ')')

def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# implementation of the depth first search ECLAT algorithm
	dataset = Dataset(filepath, vertical=True)
	nb_transactions = dataset.trans_num()
	min_support = minFrequency * nb_transactions

	eclat(dataset, min_support, nb_transactions)


def run_perf_tests():
	from time import time
	import pickle

	log = [[e for e in ["dataset", "min_freq", "apriori_time", "miner_time"]]]

	log_file = open('log2.txt', 'a') 
	print("\t\t\t     apriori() \t alternative_miner()", file=log_file)
	tested_datasets = ["mushroom.dat", "retail.dat", "pumsb.dat", "accidents.dat", "chess.dat", "connect.dat"]
	tested_frequencies = [.9, .8, .7, .6, .5, .4, .3, .2, .1]

	for iter in range(10):
		for dataset in tested_datasets:
			skip_apriori = False
			skip_miner = False
			for min_freq in tested_frequencies:
				start = time()
				if not skip_apriori: apriori("Datasets/" + dataset, min_freq)
				mid = time()
				if not skip_miner: alternative_miner("Datasets/" + dataset, min_freq)
				end = time()
				print(dataset + "\t\t" + str(min_freq) + "\t\t" + (("{:.2f}".format(mid-start)) if not skip_apriori else "N/A") + "\t\t" + (("{:.2f}".format(end-mid)) if not skip_miner else "N/A"), file=log_file)
				log.append([dataset, min_freq, mid-start, end-mid])

				if mid - start > 120 : skip_apriori = True
				if end - mid > 120 : break

	pickle_out = open("log.pickle","wb")
	pickle.dump(log, pickle_out)
	pickle_out.close()
	log_file.close() 


if __name__ == "__main__":
    # apriori("Datasets/chess.dat", 0.8)
	alternative_miner("Datasets/chess.dat", 0.8)
	# run_perf_tests()
