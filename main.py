import cv2
import numpy as np
from math import hypot

def distance(p1,p2):
	return hypot(p2[0]-p1[0],p2[1]-p1[1])

def shape(factor):
	if factor >= 0.7 and factor <= 0.8:
		return 'Circle'
	elif factor >= 0.484 and factor <= 0.55:
		return 'Square'
	elif factor >= 0.44 and factor <= 0.483:
		return 'Triangle'
	elif factor >= 0.36 and factor <= 0.38:
		return 'Diamond'
	elif factor >= 0.32 and factor <= 0.34:
		return 'Oval'
	elif factor >= 0.2 and factor <= 0.3:
		return 'Rectangle'
	else:
		return 'Shape not recognized'

def main():

	#reading the image
	im = cv2.imread('laptop.jpg')
	# im = cv2.resize(im, (400,400))
	cv2.imshow('orange', im)

	#converting the image to grayscale
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	#contrast enchancement - Contrast Limited Adaptive Histogram Equalization
	clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
	im = clahe.apply(im)

	#noise removal - Gaussian filter
	im = cv2.GaussianBlur(im, (21,21), 0)

	#convering image to binary
	im = cv2.threshold(im, 225, 255, cv2.THRESH_BINARY_INV)[1]

	#finding region boundries (countours)
	(_,cnt, _) = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imshow('orange binary', im)
	print(len(cnt))
	areatoc=cnt[0]
	area = cv2.contourArea(areatoc)
	maxDim = 0
	for i in range(0,len(areatoc)):
		for j in range(i+1, len(areatoc)):
			dist = distance(areatoc[i][0],areatoc[j][0])
			if dist > maxDim:
				maxDim = dist
	#this formula is only true for circles
	print(maxDim,area)
	diameter = maxDim
	sf = area/diameter**2
	print('Shape factor = ', sf)
	print(shape(sf))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()