import sys
import pickle
from operator import itemgetter
from train import readProbDict 

###############################################################################
#
# Name: highestProbMsg
#
# Purpose: This function generates the most typical message of a given length
#	   for a given user. The most typical message is defined to be the
#	   sequence of words given transition probabilities from a Markov
#	   model trained on bigrams.
#
# Returns: the sequence of words given transition probabilities from a Markov
#	   model trained on bigrams.
#
###############################################################################
def highestProbMsg(user, transProb, vocab, length):
	# Find all the word pairs with non zero transition probability
	nonZeroProbs = transProb.keys()
	
	# Generate the most likely word sequence
	msg = ["START"]
	for i in range(1, length + 1):
		wordProbs = []
		
		# For each word
		for word in vocab:
			# Find each transition probability
			pair = (word, msg[i - 1])
			if pair in nonZeroProbs:
				wordProbs.append((word, transProb[pair]))
			else:
				wordProbs.append((word, 0))
		
		# Sort words by their transition probabilities
		wordProbs = sorted(wordProbs, key = itemgetter(1), 
				   reverse = True)
				   
		# Add the most likely word to the message
		msg.append(wordProbs[0][0])		
	
	# Return the message without the phony start word
	return msg[1:]		

if __name__ ==  "__main__":
	# Read the transition probability matrix
	transProb = readProbDict(sys.argv[1])
	users = transProb.keys()

	# Read the average message length of each user
	avgLengths = readProbDict(sys.argv[2])
	
	# Read the vocabulary of each user
	vocabs = readProbDict(sys.argv[3])

	#
	typMsgs = {user:highestProbMsg(user, transProb[user], vocabs[user],\
		   avgLengths[user]) for user in users}
	for k in typMsgs.keys():
		print(k, ":", " ".join(typMsgs[k]))
		print()