import numpy as np


def read_file():
	with open("IrisData.txt", "r") as flowerFile:
		flowerData = np.empty([50, 5, 3])
		noSample = type = 0
		next(flowerFile)
		for line in flowerFile:
			line = line.split(",")
			line[-1] = type
			flowerData[noSample % 50, :, type] = np.array(line)
			noSample += 1
			if noSample % 50 == 0:
				type += 1
		np.random.shuffle(flowerData)
		return flowerData


############################################################################


class BackPropagation:
	def __init__(self, eta, epochs, bias, neurons, fn, data):
		self.noSampleType = len(data[:, 0, 0])
		self.noFeature = len(data[0, :-1, 0])
		self.noOut = len(data[0, 0, :])

		self.toTrain = 30  # < 50 noSampleType
		self.toTest = self.noSampleType - self.toTrain

		self.noData = self.noSampleType * self.noOut
		self.noTrain = self.toTrain * self.noOut
		self.noTest = self.toTest * self.noOut

		self.eta = eta
		self.epochs = epochs
		self.bias = bias
		self.neurons = [self.noFeature] + neurons + [self.noOut]
		self.layers = len(self.neurons)

		obj = Activation()
		self.activation = (
			[obj.sigmoid, obj.sigmoid_]
			if fn == "Sigmoid"
			else [obj.hyperbolicTangent, obj.hyperbolicTangent_]
			if fn == "Hyperbolic Tangent"
			else None
		)

		self.trainSample, self.testSample = self.divide(data)
		self.W, self.net, self.error = self.network()

		self.indexer = 0

	def train(self):
		algorithm = [self.forward, self.backward, self.update]
		[
			[[_() for _ in algorithm] for _ in range(self.noTrain)]
			for _ in range(self.epochs)
		]

	def forward(self):
		self.net[0][1:] = self.trainSample[self.indexer % self.noTrain, :-1, np.newaxis]
		for i in range(self.layers - 1):
			self.net[i + 1][1:] = self.activation[0](np.dot(self.W[i], self.net[i]))

	def backward(self):
		actual = np.zeros(self.net[-1][1:].shape)
		actual[int(self.trainSample[self.indexer % self.noTrain, -1])] = 1
		self.indexer += 1
		self.error[-1] = self.activation[1](self.net[-1][1:]) * (
			actual - self.net[-1][1:]
		)

		for i in reversed(range(self.layers - 2)):
			self.error[i] = self.activation[1](self.net[i + 1][1:]) * np.dot(
				self.W[i + 1][:, 1:].transpose(), self.error[i + 1]
			)

	def update(self):
		for i in range(self.layers - 1):
			self.W[i] += self.eta * np.dot(self.error[i], self.net[i].transpose())

	def test(self):
		success = 0
		for i in range(self.noTest):
			self.net[0][1:] = self.testSample[i, :-1, np.newaxis]
			for j in range(self.layers - 1):
				self.net[j + 1][1:] = self.activation[0](np.dot(self.W[j], self.net[j]))

			if self.testSample[i, -1] == np.argmax(self.net[-1][1:]):
				success += 1
		return round((success / self.noTest) * 100, 1)

	def divide(self, data):
		train, test = np.empty([self.noTrain, self.noFeature + 1]), np.empty(
			[self.noTest, self.noFeature + 1]
		)
		for i in range(self.toTrain):
			for j in range(self.noOut):
				train[i * self.noOut + j] = data[i, :, j]

		for i in range(self.toTest):
			for j in range(self.noOut):
				test[i * self.noOut + j] = data[i + self.toTrain, :, j]

		np.random.shuffle(train)
		np.random.shuffle(test)
		return train, test

	def network(self):
		W, net, error = ([] for _ in range(3))
		for i in range(self.layers - 1):
			W.append(
				np.insert(
					np.random.rand(self.neurons[i + 1], self.neurons[i])
					/ np.sqrt(self.neurons[i] + 1),
					0,
					self.bias,
					axis=1,
				)
			)
			error.append(np.empty([self.neurons[i + 1], 1]))

		for i in range(self.layers):
			net.append(np.insert(np.empty([self.neurons[i], 1]), 0, self.bias, axis=0))
		return W, net, error


############################################################################


class Activation:
	def __init__(self):
		self.a = 1  # > 0  slope of sigmoid
		self.b = 1  # > 0  case tanh

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-self.a * x))

	def sigmoid_(self, x):
		return self.a * x * (1 - x)

	def hyperbolicTangent(self, x):
		return self.a * np.tanh(x * self.b)

	def hyperbolicTangent_(self, x):
		return (self.b / self.a) * (self.a - x) * (self.a + x)
