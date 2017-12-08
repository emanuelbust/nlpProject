import numpy as np
import sys
import utils
import dynet_config
import dynet as dy
from os import path
from operator import itemgetter

# I don't know what this is
dynet_config.set(
    mem=4096,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)

###############################################################################
#
# Name: readTrainingData
#
# Purpose: 
#
# Returns: A list with three elements. The first is the messages grouped by
#	   word, the second is the person who sent the message, once for 
#	   each word in the message, and all of the possible labels. 
#
###############################################################################
def readTrainingData(path):
	lines = [line.replace("\n", "").split("\t") for line in open(path, "r")]
	messages = [line[2] for line in lines]
	labels = [line[0] for line in lines]
	uniqueLabels = uniqueList(labels)

	# Match the dimension of the label matches the dimenstion of the message	
	for i in range(len(labels)):
		labels[i] = [labels[i]] * len(messages[i].split())

	# Split the message into words
	messages = [message.split() for message in messages]

	return [messages, labels, uniqueLabels]

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

def getVocab(messages):
	allWords = []
	for message in messages:
		for word in message:
			allWords.append(word)
	
	return uniqueList(allWords)

def wordsToIndices(wordSeq, wordToIndex):
	indexSeq = []
	
	# For each word
	for word in wordSeq:
		# He uncapitalizes the word here, but you're not
		# Look up the index and acumulate it
		index = wordToIndex.get(word, 0)
		indexSeq.append(index)
	
	return indexSeq

def forwardPass(message):
	# Map the words in a message to its word embeddings
	inputSeq = [embeddingParameters[word] for word in message]
	
	# Convert parameters to expressions???????????
	w = dy.parameter(projectionWeight)
	b = dy.parameter(projectionBias)

	# Initialize the RNN unit?????????
	rnnSeq = rnnUnit.initial_state()
	
	# Run each embedding through the RNN
	rnnHiddenOuts = rnnSeq.transduce(inputSeq)
	
	# Project each words output to the size of labels
	rnnOutputs = [dy.transpose(w) * h + b for h in rnnHiddenOuts]

	return rnnOutputs

def predict(outputList):
	# Take the softmax of each output
	predProbs = [dy.softmax(output) for output in outputList]
	
	# Make each softmax a numpy value
	predProbsNP = [prob.npvalue() for prob in predProbs]

	# Find the argmax for each output
	predProbIndex = [np.argmax(x) for x in predProbsNP]

	return predProbIndex

def checkScore(prediction, true):
	return 1 if prediction == true  else 0

def getAccuracy(scores):
	return float(sum(scores) / len(scores))

def evaluate(nestedPreds, nestedTrue):
	flatScores = []

	# For each set of message labels
	for i in range(len(nestedTrue)):
		scores = []
		pred = nestedPreds[i]
		true = nestedTrue[i]
		
		# For each word
		for p,t in zip(pred, true):
			score = checkScore(p, t)
			scores.append(score)
		flatScores.extend(scores)

	# Calculate correctness
	accuracy = getAccuracy(flatScores)

	return accuracy

def train():
	epochLosses = []
	overallAccuracies = []

	for i in range(NUM_EPOCHS):
		# Shuffle the training data
		np.random.seed(i)
		np.random.shuffle(trainingMessages)

		# Shuffle the training labels
		np.random.seed(i)
		np.random.shuffle(trainingLabels)

		epochLoss = []
		for j in range(NUM_BATCHES_TRAINING):
			# Start with a clean computation graph
			dy.renew_cg()

			# Build the batch
			batchTokens = trainingMessages[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
			batchLabels = trainingLabels[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

			# Iterate through the batch
			for k in range(len(batchTokens)):
				# Map words to indices
				indexSeq = wordsToIndices(batchTokens[k], wordToIndex)
				
				# Pass through the network
				predictions = forwardPass(indexSeq)

				# Calculate the loss of each token in each sample
				loss = [dy.pickneglogsoftmax(predictions[l], batchLabels[k][l]) for l in range(len(predictions))]
			
				# Sum the loss of each word
				msgLoss = dy.esum(loss)

				# Backpropogate the loss
				msgLoss.backward()
				TRAINER.update()
				epochLoss.append(msgLoss.npvalue())

				
				# Check the prediction of sample sentence
			if j % 250 == 0:
				print("epoch {}, batch {}".format(i+1, j+1))
				samp = forwardPass(wordsToIndices(sample, wordToIndex))
				predictions = [indexToLabel[p] for p in predict(samp)]				
				print(list(zip(sample, predictions)))
					
		# Record epoch loss
		epochLosses.append(np.sum(epochLoss))
		print("testing after epoch {}".format(i+1))
		epochPredictions = test()
		epochOverallAccuracy = evaluate(epochPredictions, testingLabels)
		overallAccuracies.append(epochOverallAccuracy)
		
		print("Loss", np.sum(epochLoss))	
		print("Accuracy", realTest(finalLayer(test()), testingLabels))

	print(overallAccuracies)
			
	return epochLosses, overallAccuracies

def test():
	allPredictions = []
	
	# For each batch
	for j in range(NUM_BATCHES_TESTING):
		# Start with a clean computation graph
		dy.renew_cg();

		# Make the batch
		batchTokens = testingMessages[j*BATCH_SIZE:(j+1)*BATCH_SIZE]

		# For each point in the batch
		for k in range(len(batchTokens)):
			# Map words to emedding indices
			indexSeq = wordsToIndices(batchTokens[k], wordToIndex)
			# Go through the RNN
			preds = forwardPass(indexSeq)
			# Form labels
			labelPreds = predict(preds)
			allPredictions.append(labelPreds)
	
	return allPredictions

def finalLayer(predictions):
	msgPred = []
	for pred in predictions:
		counts = [(x, pred.count(x)) for x in pred]
		counts = sorted(counts, key = itemgetter(1), reverse = True)
		msgPred.append(counts[0][0])
	return msgPred

def realTest(predictions, labels):
	count = 0.0
	for i in range(len(predictions)):
		if predictions[i] in labels[i]:
			count += 1
	return count / len(predictions)

if __name__ == "__main__":
	# Read training data (points and labels)
	trainingData = readTrainingData(sys.argv[1])
	trainingMessages = trainingData[0]
	trainingLabels = trainingData[1]
	labels = trainingData[2]

	# Read testing data (points and labels)
	testData = readTrainingData(sys.argv[2])
	testingMessages = testData[0]
	testingLabels = testData[1]

	# Map labels to indices and vice veresa
	labelToIndex = {labels[i]:i for i in range(len(labels))}
	indexToLabel = {val:key for key, val in labelToIndex.items()}

	# Convert labels to integers using the map
	trainingLabels = [[labelToIndex[label] for label in message] for message in trainingLabels]
	testingLabels = [[labelToIndex[label] for label in message] for message in testingLabels]

	# Initialze an empty model
	rnnModel = dy.ParameterCollection()

	# Initialize hyper parameters
	EMBEDDING_DIM = 300
	HIDDEN_SIZE = 200
	NUM_LAYERS = 1

	# Map word in the vocabulary and to indices
	vocab = getVocab(trainingMessages)
	wordToIndex = {vocab[i]:i for i in range(len(vocab))}

	# Add random embeddings
	embeddingParameters = rnnModel.add_lookup_parameters((len(wordToIndex) + 1, EMBEDDING_DIM))

	# Add RNN
	rnnUnit = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_DIM, HIDDEN_SIZE, rnnModel)

	# Add a projection layer weights
	projectionWeight = rnnModel.add_parameters((HIDDEN_SIZE, len(list(labelToIndex.keys()))))

	# Add projection layer bias
	projectionBias = rnnModel.add_parameters((len(list(labelToIndex.keys()))))
	
	'''
	# Sucessfully ran this through the RNN
	sample = "Down as shit motherfuckers".split()
	sample = wordsToIndices(sample, wordToIndex)
	sample = forwardPass(sample)
	print(sample[0].npvalue())
	'''
	sample = "Down as shit motherfuckers Scott".split()

	'''
	# Sucessfully predicct labels
	sent = "Down as shit motherfuckers".split()
	sample = wordsToIndices(sent, wordToIndex)
	sample = forwardPass(sample)
	predict = predict(sample)
	predict = [indexToLabel[label] for label in predict]	
	print(list(zip(sent, predict)))
	'''

	# Initialize a trainer
	TRAINER = dy.SimpleSGDTrainer(m = rnnModel, learning_rate = .01)
	
	# Set parameters for batching
	BATCH_SIZE = 256
	NUM_BATCHES_TRAINING = int(np.ceil(len(trainingMessages) / BATCH_SIZE)) 
	NUM_BATCHES_TESTING = int(np.ceil(len(testingMessages) / BATCH_SIZE)) 

	# Defined train here

	# Pick number of epochs
	NUM_EPOCHS = 100

	# Defined checkScore and getAccuracy here

	train()
