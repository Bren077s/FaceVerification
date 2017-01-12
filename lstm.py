# Stacked LSTM for international airline passengers problem with memory
import numpy
import pandas
import math
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
import sys

fit = "false"
modfile = "model.json"
weightfile = "weights.hdf5"
if(len(sys.argv) >= 2):
	if(sys.argv[1] == "true"):
		fit = "true"
	else:
		modfile = sys.argv[1]
if(len(sys.argv) >= 3):
	weightfile = sys.argv[2]

class ModelReset(Callback):
	def on_epoch_end(self, epoch, logs={}):
		self.model.reset_states()
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = pandas.read_csv('goog_open_raw.csv', usecols=[0], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 50
batch_size = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))


if(fit == "true"):
	# create and fit the LSTM network
	model = Sequential()
	#model.add(LSTM(256, batch_input_shape=(batch_size, look_back, 1), return_sequences=True, stateful=True))
	model.add(LSTM(256,batch_input_shape=(batch_size, look_back, 1), stateful=True))
	model.add(Dense(1))
	model.add(Activation('linear'))
	learning_rate = 0.0001
	epochs = 1000
	decay_rate = learning_rate / epochs
	adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
	model.compile(loss='mean_squared_error', optimizer=adam)
	reset = ModelReset()
	filepath="weights-improvement-{loss}-{epoch:03d}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')
	
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	
	model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, shuffle=False,callbacks=[reset, checkpoint])

	# serialize weights to HDF5
	model.save_weights("model.hdf5")
	print("Saved model to disk")
else:
	# later...
	 
	# load json and create model
	json_file = open('{0}'.format(modfile), 'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	# load weights into new model
	model.load_weights("{0}".format(weightfile))
	print("Loaded model from disk")


	# make predictions
	model.reset_states()
	trainPredict = model.predict(trainX, batch_size=batch_size)
	#model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)
	data = numpy.array(testPredict[-50:])
	data = numpy.reshape(data, (1, 50, 1))

	output = []
	for i in range(300):
		futurePredict = model.predict(data, batch_size=batch_size)
		output.append(futurePredict[0,0])
		#print(data)
		data[0,:-1] = data[0,-49:]
		data[0,49] = output[i]
	futurePredict = numpy.array(output)
	futurePredict = numpy.reshape(futurePredict, (futurePredict.shape[0],1))
	trainY = numpy.array(trainY)
	testY = numpy.array(testY)
	trainY = numpy.reshape(trainY, (1, trainY.shape[0]))
	testY = numpy.reshape(testY, (1, testY.shape[0]))
	# invert predictions
	#trainPredict = scaler.inverse_transform(trainPredict)
	#trainY = scaler.inverse_transform(trainY)
	#testPredict = scaler.inverse_transform(testPredict)
	#testY = scaler.inverse_transform(testY)
	#futurePredict = scaler.inverse_transform(futurePredict)
	
	# calculate root mean squared error
	trainScore = mean_squared_error(trainY[0], trainPredict[:,0])
	print('Train Score: %.2f MSE' % (trainScore))
	testScore = mean_squared_error(testY[0], testPredict[:,0])
	print('Test Score: %.2f MSE' % (testScore))
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty([dataset.shape[0]+futurePredict.shape[0], dataset.shape[1]])
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = numpy.empty([dataset.shape[0]+futurePredict.shape[0], dataset.shape[1]])
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# shift test predictions for plotting
	futurePredictPlot = numpy.empty([dataset.shape[0]+futurePredict.shape[0], dataset.shape[1]])
	futurePredictPlot[:, :] = numpy.nan
	futurePredictPlot[len(dataset):len(dataset)+futurePredict.shape[0], :] = futurePredict
	# plot baseline and predictions
	import matplotlib.pyplot as plt
	plt.plot(dataset)
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.plot(futurePredictPlot)
	plt.show()