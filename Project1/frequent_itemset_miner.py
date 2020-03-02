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
from copy import copy

class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath, vertical=False):
		"""reads the dataset file and initializes files"""

		self.transactions = list()
		self.items = {}
		self.items_appearance = {}
		self.projection = []

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			i = 0
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.transactions.append(transaction)
				for item in transaction:
					# self.items.add(item)
					if not vertical:
						if (item,) not in self.items:
							self.items[(item,)] = 1
						else:
							self.items[(item,)] +=1
					else:
						if item not in self.items_appearance:
							self.items_appearance[item] = [i]
						else:
							self.items_appearance[item].append(i)
				i += 1
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


def transaction_includes_itemset(transaction, itemset): # TODO time-consuming
	for item in itemset:
		if item not in transaction:
			return False
	return True

def print_frequent_itemsets(level, dataset):
	for itemset in level.keys():
		if level[itemset] is not None:
			print(str(list(itemset)) + ' (' + str(level[itemset]/dataset.trans_num()) + ')')

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
			if len(new_itemset) > len(itemset1) + 1: continue # TODO should break if itemsets are sorted
			yield new_itemset

def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	dataset = Dataset(filepath, vertical=False)

	#prev_level = dict.fromkeys({tuple([x]) for x in dataset.get_items()}, 0)
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
		min_support = minFrequency * dataset.trans_num()
		for itemset in prev_level.keys():
			if prev_level[itemset] < min_support:
				prev_level[itemset] = None
		print_frequent_itemsets(prev_level, dataset)

		# GENERATE CANDIDATES FOR NEXT LEVEL
		next_level = dict.fromkeys({tuple(x) for x in combine_frequent_itemsets(prev_level)}, 0)
		prev_level = next_level

def print_frequent_itemset_from_projected_database(dataset, item, nb_transactions):
	itemset = copy(dataset.projection)
	itemset.append(item)
	print(str(itemset) + ' (' + str(len(dataset.items_appearance[item])/nb_transactions) + ')')

def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	dataset = Dataset(filepath, vertical=True)
	min_support = minFrequency * dataset.trans_num()

	projected_database = {}
	for item in dataset.items_appearance:
		if len(dataset.items_appearance[item]) > min_support:
			projected_database
			print_frequent_itemset_from_projected_database(dataset, item, dataset.trans_num())


	itemset = dataset.get_first_level()
	projectedD = None

	DFS(itemset, projectedD)


def DFS(itemset, projectedD):
	pass





def run_perf_tests():
	from time import time

	log = "\t\t\t     apriori() \t alternative_miner()\n"
	tested_datasets = ["accidents.dat", "chess.dat", "mushroom.dat", "retail.dat"]
	tested_frequencies = [.9, .8, .7, .6, .5, .4, .3, .2, .1]

	for dataset in tested_datasets:
		for min_freq in tested_frequencies:
			start = time()
			apriori("Datasets/" + dataset, min_freq)
			mid = time()
			alternative_miner("Datasets/" + dataset, min_freq)
			end = time()
			log += dataset + "\t\t" + str(min_freq) + "\t\t" + "{:.2f}".format(mid-start) + "\t\t" + "{:.2f}".format(end-mid) + "\n"
			
			if mid - start > 80 : break
	
	log_file = open('log.txt', 'a') 
	print(log, file=log_file)
	log_file.close() 


# apriori("Datasets/connect.dat", 0.9)
alternative_miner("Datasets/toy.dat", 0.125)
# run_perf_tests()
