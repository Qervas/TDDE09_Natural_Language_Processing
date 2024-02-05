# Skeleton code for lab L2 (advanced)

# Do not use any imports other than the ones already in this file.

import numpy as np


# The next four functions are adapted from cbow-classifier.ipynb


def load_data(filename, max_length=20):
	items = []
	with open(filename, 'rt', encoding='utf-8') as fp:
		for line in fp:
			sentence, label = line.rstrip().split('\t')
			items.append((sentence.split()[:max_length], int(label)))
	return items


def make_vocab(data):
	vocab = {'<pad>': 0}
	for sentence, label in data:
		for t in sentence:
			if t not in vocab:
				vocab[t] = len(vocab)
	return vocab


def vectorize(vocab, data):
	max_length = max(len(s) for s, _ in data)
	xs = []
	ys = []
	for sentence, label in data:
		x = [0] * max_length
		for i, w in enumerate(sentence):
			if w in vocab:
				x[i] = vocab[w]
		xs.append(x)
		ys.append(label)
	return np.array(xs), np.array(ys)


def accuracy(y_pred, y):
	return np.mean((y_pred == y).astype(float))


def minibatches(x, y, batch_size):
	random_indices = np.random.permutation(x.shape[0])
	for i in range(0, x.shape[0] - batch_size + 1, batch_size):
		batch_indices = random_indices[i:i+batch_size]
		yield x[batch_indices], y[batch_indices]


# Efficient implementation of the softmax function; see
# http://stackoverflow.com/a/39558290

def softmax(z):
	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis] # necessary step to do broadcasting
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div = div[:, np.newaxis] # dito
	return e_x / div


# Your code below

class CBOW(object):
	def __init__(self, num_embeddings, output_dim, embedding_dim=30):
		# Initialize parameters E, W, b
		scale_W = np.sqrt(1.0 / embedding_dim)
		self.E = np.random.normal(0, 0.1, (num_embeddings, embedding_dim))
		self.W = np.random.uniform(-scale_W, scale_W, (embedding_dim, output_dim))
		self.b = np.random.uniform(-scale_W, scale_W, (1, output_dim))

	def forward(self, X):
		# Average embeddings for context
		avg_E = np.mean(self.E[X], axis=1)  # Assuming X is already one-hot-encoded
		# Linear transformation
		z = np.dot(avg_E, self.W) + self.b
		# Softmax
		return softmax(z)

	def backward(self, X, y_true, lr, l2_penalty=0.001):
		# Forward pass
		predictions = self.forward(X)
		# One-hot encode y
		y_true_one_hot = np.eye(self.W.shape[1])[y_true]
		# Compute error delta_1
		delta_1 = predictions - y_true_one_hot
		# Average embedding for context
		avg_E = np.mean(self.E[X], axis=1)
		# Update W and b
		grad_W = np.dot(avg_E.T, delta_1) / X.shape[0]
		grad_b = np.dot(np.ones((1, X.shape[0])), delta_1) / X.shape[0]
		self.W -= lr * (grad_W + 2 * l2_penalty * self.W)
		self.b -= lr * (grad_b + 2 * l2_penalty * self.b)
		# Update embeddings
		grad_E = np.dot(delta_1, self.W.T) / X.shape[1]
		# Assuming X is not one-hot-encoded, need to sum the gradients for all instances where each word appears
		for i, word_idx in enumerate(X.flatten()):
			self.E[word_idx] -= lr * grad_E[i // X.shape[1]]


def train(vocab, train_x, train_y, n_epochs=30, batch_size=24, lr=0.1):
	model = CBOW(len(vocab), np.max(train_y) + 1, 30)
	for epoch in range(n_epochs):
		# Learning rate decay
		lr *= 0.95  # Reduce the learning rate by 5% each epoch
		for X_batch, y_batch in minibatches(train_x, train_y, batch_size):
			model.backward(X_batch, y_batch, lr)
		print('Epoch {}: lr={:.4f}'.format(epoch + 1, lr))
	return model


# MAIN METHOD


def main():
	import sys

	train_data = load_data('sst-5-train.txt')
	dev_data = load_data('sst-5-dev.txt')

	vocab = make_vocab(train_data)
	train_x, train_y = vectorize(vocab, train_data)
	dev_x, dev_y = vectorize(vocab, dev_data)

	accuracies = []
	for _ in range(10):
		model = train(vocab, train_x, train_y)
		dev_output = model.forward(dev_x)
		dev_y_pred = np.argmax(dev_output, axis=1)
		accuracies.append(accuracy(dev_y_pred, dev_y))
	print('Average accuracy: {:.4f}'.format(sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
	main()
