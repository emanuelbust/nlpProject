import pickle
import sys

def read(path):
	lookUp = {}
	for line in open(path, "r"):
		entries = line.replace("\n", "")
		entries = entries.split()
		word = entries[0]
		embedding = [float(num) for num in entries[1:]]
		lookUp[word] = embedding

	return lookUp

if __name__ == "__main__":
	inFile = sys.argv[1]
	lookUp = read(inFile)
	pickle.dump(lookUp, open("embeddings.bin", "wb"))
