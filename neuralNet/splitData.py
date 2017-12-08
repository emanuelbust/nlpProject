import sys
import random
import numpy as np

def parsePoints(path):
	return [line for line in open(path, "r")]

def splitInThirds(points):
	firstThird = []
	secondThird = []
	thirdThird = []
	for point in points:
		ind = random.random()
		if ind < 1 / 3.0:
			firstThird.append(point)
		elif ind < 2 / 3.0:
			secondThird.append(point)
		else:
			thirdThird.append(point)

	return [firstThird, secondThird, thirdThird]

def write(points, path):
	with open(path, "w") as outFile:
		for point in points:
			outFile.write(point)

if __name__ == "__main__":
	# Read the data and split it
	points = parsePoints(sys.argv[1])
	splits = splitInThirds(points)

	# Write the splits to data
	write(splits[0], "train.txt")
	write(splits[1], "dev.txt")
	write(splits[2], "test.txt")
