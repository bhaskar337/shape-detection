import cv2
import numpy as np


def main():

	#reading the image
	im = cv2.imread('orange.jpg')
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
	im = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY_INV)[1]

	#finding region boundries (countours)
	(_,cnt, _) = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.imshow('orange binary', im)

	area = cv2.contourArea(cnt[0])

	#this formula is only true for circles
	diameter = np.sqrt(4*area/np.pi)

	print('Shape factor = ', area/diameter**2)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()