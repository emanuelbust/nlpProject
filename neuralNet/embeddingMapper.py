import pickle
import sys

# Read the data amd split it in thirds
def read(path):
	lookUp = {}
	for line in open(path, "r"):
		# Parse the word and vector
		entries = line.replace("\n", "")
		entries = entries.split()
		word = entries[0]
		embedding = [float(num) for num in entries[1:]]
		
		# Map the word to its embedding
		lookUp[word] = embedding

	return lookUp

if __name__ == "__main__":
	inFile = sys.argv[1]
	lookUp = read(inFile)

	# Write the mapping to a file
	pickle.dump(lookUp, open("embeddings.bin", "wb"))
