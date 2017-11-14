import argparse
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.models import load_model

def Lanet():
	"""
	Lanet model with Normalization and Cropping.
	"""
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((50, 20), (0, 0))))
	model.add(Convolution2D(6, 5, 5, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(16, 5, 5, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(84, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model

def Nvidia():
	"""
	Model from Nvidia with Normalization and Cropping.
	"""
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((50, 20), (0, 0))))
	model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))
	model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))
	model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
	model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1,1)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(.5))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(.5))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1))
	return model

def generator(samples, batch_size=32):
	"""
	generator for the samples with 3 cameras and Augmentation.
	"""
	num_samples = len(samples)
	
	while 1: 		# Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					#file_name = batch_sample[i].split('\\')[-1]
					file_name = batch_sample[i].split('/')[-1]
					file_path = args.dataset + '/IMG/' + file_name
					image = cv2.imread(file_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # cv2.imread get the image in BGR mode.
					aug_image = cv2.flip(image, 1)
					images.append(image)
					images.append(aug_image)

				center_angle = float(batch_sample[3])
				left_angle = center_angle + args.correction
				right_angle = center_angle - args.correction
				angles.append(center_angle)
				angles.append(center_angle * (- 1.0))   # angle of augmentated center image
				angles.append(left_angle)
				angles.append(left_angle * (- 1.0))     # angle of augmentated left image
				angles.append(right_angle)
				angles.append(right_angle * (- 1.0))    # angle of augmentated right image

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

def predict_image():
	"""
	predict on 2 images to check the model training result.
	"""
	images = []

	image = cv2.imread('./data0/IMG/center_2016_12_01_13_32_52_551.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images.append(image)
	
	image = cv2.imread('./data0/IMG/center_2016_12_01_13_32_43_761.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images.append(image)

	X_predict = np.array(images)

	print("________Expected predict: ", -0.23, 0.367)

	return X_predict

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Bahavioral Cloning Model')
	parser.add_argument(
		'action',
		type=str,
		help='To train or predict.'
	)
	parser.add_argument(
		'--dataset',
		type=str,
		default='./data0',
		help='Path to train dataset, dataset should be on the same path.'
	)
	parser.add_argument(
		'--correction',
		type=float,
		default=0.1,
		help='Correction of steering angle for the left and right cameras.'
	)
	args = parser.parse_args()

	# 0 Read the the image path and steering info from csv file 
	lines = []
	with open(args.dataset + '/driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
	        lines.append(line)

	# split the validation set from training set
	shuffle(lines)
	train_lines, validation_lines = train_test_split(lines, test_size=0.2)
	print("++++++++Num of train_lines, validation_lines", len(train_lines), len(validation_lines)) 

	# 1 Using the generator function
	train_generator = generator(train_lines, batch_size=32)
	validation_generator = generator(validation_lines, batch_size=32)

	# 2 Build the model
	model = Nvidia()

	if args.action == 'train':
		#train the model.
		model.compile(optimizer='adam', loss='mse')
		early_stopping = EarlyStopping(monitor='val_loss', patience=1)
		model.fit_generator(train_generator, samples_per_epoch= len(train_lines) * 6, 
			validation_data=validation_generator, nb_val_samples=len(validation_lines) * 6, nb_epoch=3, callbacks=[early_stopping])
		
		# 4 Save the model
		model.save('model.h5')
	elif args.action == 'predict':
		#make prediction
		model = load_model('model.h5')
		y_predict = model.predict(predict_image(), batch_size = 1)
		print("_________predict: ", y_predict)