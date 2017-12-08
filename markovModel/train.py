import sys
import pickle
from itertools import product
from collections import Counter
from nltk.tokenize import TweetTokenizer

###############################################################################
#
# Name: parseInput
#
# Purpose: This function takes a file path and reads the file. Lines who have
#	   the strings delimited by a tab are saved into an array. Then
#	   the tab delineated entries are stored in a tuple with the last
#	   entry being tokenized.
#
# Returns: A list of tuples corresponding to lines in the file with three tab
#	   delineated entries. The last entry of the line is tokenized and
#	   stored as a list.
#
###############################################################################
def parseInput(path):
	# Make a tokenizer
	tokenizer = TweetTokenizer()

	# Read each line and split by tab
	data = [line.replace("\n", "").split("\t") for line in open(path, "r")]
	
	# Filter out those who have an empty message
	data = [point for point in data if len(point) == 3]
	
	# Tokenize the message
	data = [(point[0], point[1], tuple(tokenizer.tokenize(point[2])))\
		for point in data]	

	return data

###############################################################################
#
# Name: uniqueList
#
# Purpose: This function takes removes duplicates from a list. This is done
#	   by casting the list to a set, which don't have repeats, and then
#	   casting the set back into a list.
#
# Returns: A list whose contains all of the elements in the given list 
# 	   exactly once.
#
###############################################################################
def uniqueList(givenList):
	# Make a set from the list
	listAsSet = frozenset(givenList)

	# Make a list from the set
	uniqueElements = list(listAsSet)

	return uniqueElements
	
###############################################################################
#
# Name: generateTransProb
#
# Parameters: wordsByMessage - a sequence of words as they appear in the data
#			       grouped by message.
#	      allWords - a list of unique tags observed in the data
#
# Purpose: This function generates a map from pairs of words to their
#	   conditional probabilities. More specifically, this function returns
#	   a dictionary mapping word pair (x, y) -> (x|y). p(x|y) in defined to
#	   be the frequency of the pair (y, x) is the data over the frequency
#	   of y in the data. If the sequence (x,y) never occurs, then it is
#	   not entered into the dictionary.
#
# Returns: A dictionary mapping word pair (x, y) to p(x|y) where (x, y) is each
#	   pair of words from the unique words given by the user. All pairs
#	   except those that never appear appear in the dictionary.
#
###############################################################################
def generateTransProb(wordsByMessage, allWords, wordCounts):
	# Generate all adjacent tags and then count them
	wordPairSeq = []
	for wordSequence in wordsByMessage:
		for i in range(1, len(wordSequence)):
			wordPairSeq.append((wordSequence[i - 1], wordSequence[i]))
	wordPairCounts = Counter(wordPairSeq)
				
	# Calculate and store each transition probability
	transProbDict = {}
	for wordOne in allWords:
		for wordTwo in allWords:
			if wordPairCounts[(wordTwo, wordOne)] != 0:
				transProbDict[(wordOne, wordTwo)] = 0.0 + \
				(wordPairCounts[(wordTwo, wordOne)] /\
				wordCounts[wordTwo])
	return transProbDict		
	
###############################################################################
#
# Name: writeProbDict
#
# Parameters: probDict -  the dictionary that maps tuples to conditional 
#			  probabilities
#	      filePath - the file path where the dictionary will be written
#
# Purpose: This function writes the the emission and transition dictionaries
#	   to their respective files
#
# Returns: Nothing
#
###############################################################################
def writeProbDict(probDict, filePath):
	with open(filePath, "wb") as outFile:
		pickle.dump(probDict, outFile)

###############################################################################
#
# Name: readProbDict
#
# Parameters: filePath - the file path where the transition probability 
#			 dictionary was written
#
# Purpose: This function reads the probability dictionary from a given file. 
#
# Returns: A reference to the pickled item in the file path given.
#
###############################################################################
def readProbDict(filePath):
	with open(filePath, "rb") as inFile:
		return pickle.load(inFile)

###############################################################################
#
# Name: getTransProb
#
# Purpose: Given a user as a string and messages as a list, this function 
#	   generates a transition probability mapping word pairs in the users 
#	   vocabulary to transition probabilities between words.
#
# Returns: A map from word pairs in the users vocabulary to the transition 
#	   probability of those words. 
#
###############################################################################
def getTransProb(person, data):
	# Subset the data 
	subset = [point for point in data if point[0] == person]
	messages = [tuple(["START"] + list(point[2])) for point in subset]
	words = []
	for message in messages:
		for word in message:
			words.append(word)
	vocab = uniqueList(words)
	 		
	
	return generateTransProb(messages, vocab, Counter(words))
	
###############################################################################
#
# Name: getAverageLength
#
# Purpose: This function finds the average length of a message given a user. 
#	   This is done by subsetting the messages that were sent by the user,
#	   and then calculating their average length. 
#
# Returns: The average length of a user's messages rounded down to the nearest
#	   integer.
#
###############################################################################
def getAverageLength(person, data):
	# Subset the data 
	subset = [point for point in data if point[0] == person]
	messages = [point[2] for point in subset]
	lengths = [len(message) for message in messages]
	
	return int(sum(lengths) * 1.0 / len(lengths))

###############################################################################
#
# Name: getVocab
#
# Purpose: This function finds the vocabulary of a given user. This is done by
#	   collecting wach word of each message from the user into a list and
#	   then getting rid of repeated words. 
#
# Returns: A list of all the unique words that occur in the user's messages
#	   in o particular order is returned.
#
###############################################################################
def getVocab(person, data):
	# Subset the data 
	subset = [point for point in data if point[0] == person]
	messages = [tuple(["START"] + list(point[2])) for point in subset]
	words = []
	for message in messages:
		for word in message:
			words.append(word)
	return uniqueList(words) 

if __name__ == "__main__":
	# Parse the data get the users
	data = parseInput(sys.argv[1])
	people = uniqueList([point[0] for point in data])

	# Find the average message length
	lengths = [len(point[2]) for point in data]
	average = int(sum(lengths) * 1.0 / len(lengths))
	
	# Generate transition probabilities for each user and write them
	transMaps = {person:getTransProb(person, data) for person in people}
	outfile = "trans.bin"
	writeProbDict(transMaps, outfile)
	
	# Generate average lengths for each user and write them
	avgLengths = {p:getAverageLength(p, data) for p in people}
	outfile = "lengths.bin"
	writeProbDict(avgLengths, outfile)
	
	# Generate vocabularies and write them
	vocabs = {p:getVocab(p, data) for p in people}
	outfile = "vocabs.bin"
	writeProbDict(vocabs, outfile)