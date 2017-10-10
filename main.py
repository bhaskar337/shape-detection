import cv2
import numpy as np
from math import hypot
import sys


class shapeDetection:

	def __init__(self, show):
		self.show = show


	def _preProcess(self, im):		
		if self.show:
			cv2.imshow('orignal', im)

		#converting the image to grayscale
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		if self.show:
			cv2.imshow('grayscale', im)

		#contrast enchancement - Contrast Limited Adaptive Histogram Equalization
		clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
		im = clahe.apply(im)
		if self.show:
			cv2.imshow('CLAHE', im)

		#noise removal - Gaussian filter
		im = cv2.GaussianBlur(im, (21,21), 10)
		if self.show:
			cv2.imshow('gaussian blur', im)

		#convering image to binary
		im = cv2.threshold(im, 225, 255, cv2.THRESH_BINARY_INV)[1]
		if self.show:
			cv2.imshow('binary', im)

		return im


	def _distance(self, p1, p2):
		return hypot(p2[0]-p1[0], p2[1]-p1[1])


	def _calcDiameter(self,  areatoc):
		maxDim = 0
		for i in range(0, len(areatoc)):
			for j in range(i+1, len(areatoc)):
				dist = self._distance(areatoc[i][0], areatoc[j][0])
				if dist > maxDim:
					maxDim = dist

		return maxDim


	def _shape(self, sf):

		if sf >= 0.7 and sf <= 0.85:
			return 'Circle'
		elif sf <= 0.55:
			if sf >= 0.48:
				return 'Square'
			else:
				return 'Rectangle'
		else:
			return 'Shape not recognized'


	def detect(self, im):

		im = self._preProcess(im)
		(_,cnt, _) = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areatoc = cnt[0]
		area = cv2.contourArea(areatoc)
		diameter = self._calcDiameter(areatoc)

		sf = area/diameter**2
		return self._shape(sf)


def main():

	im = sys.argv[1]
	im = cv2.imread('images/' + im)

	if len(sys.argv) > 2:
		show = True
	else:
		show = False

	sd = shapeDetection(show)
	print('Detected Shape: ' + sd.detect(im))

	if show:
		while True:
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

if __name__ == '__main__':
	main()