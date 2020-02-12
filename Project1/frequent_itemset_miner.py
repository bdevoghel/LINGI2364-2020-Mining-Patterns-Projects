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


class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath):
		"""reads the dataset file and initializes files"""
		self.transactions = list()
		self.items = {}

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.transactions.append(transaction)
				for item in transaction:
					# self.items.add(item)
					if (item,) not in self.items:
						self.items[(item,)] = 1
					else:
						self.items[(item,)] +=1
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
	dataset = Dataset(filepath)

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

def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	apriori(filepath, minFrequency)

# apriori("Datasets/test.dat", 0.7)