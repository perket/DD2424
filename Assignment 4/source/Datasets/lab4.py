import numpy as np
from random import uniform
import matplotlib.pyplot as plt
import math
import test_data as td
import codecs

class DataLoader:

	def __init__(self, file_name):
		# Input data and unique charachters
		self.book = codecs.open(file_name, "r", "utf8").read().replace(u"\u2022", u"")
		#print(type(self.book))
		#self.book = open(file_name, 'r').read()
		#self.uniq_chars = list(set(self.book))
		self.uniq_chars = list([u'\t', u'\n', u' ', u'!', u'"', u"'", u'(', u')', u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'6', u'7', u'9', u':', u';', u'?', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'^', u'_', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'}', u'\xfc'])
		
		# Size of the book and vocabulary size
		self.book_size = len(self.book)
		self.vocab_size = len(self.uniq_chars)

		# Mappings from char to ind and vice versa
		self.char_to_ix = { ch:i for i,ch in enumerate(self.uniq_chars) }
		self.ix_to_char = { i:ch for i,ch in enumerate(self.uniq_chars) }



class RNN:

	def __init__(self, hidden_dim, learning_rate, input_size, output_size):
		# Length of input and output sequences, same for this exercise
		self.input_size = input_size
		self.output_size = output_size

		# Learning rate, eta
		self.learning_rate = learning_rate

		# Model parameters

		# hidden state at time t of size m x 1
		self.h = np.zeros((hidden_dim, 1))

		# weight matrix of size m x m applied to ht-1 (hidden-to-hidden connection)
		self.W = np.array(td.testW)#
		W = np.random.randn(hidden_dim, hidden_dim) * 0.01
		#print(self.W.shape)
		#print(W.shape)
		# weight matrix of size m x d applied to xt (input-to-hidden connection)
		self.U = np.array(td.testU)#
		U = np.random.randn(hidden_dim, self.input_size) * 0.01
		#print(self.U.shape)
		#print(U.shape)
		# weight matrix of size C x m applied to at (hidden-to-output connection)
		self.V = np.array(td.testV)#np.random.randn(self.output_size, hidden_dim) * 0.01
		V = np.random.randn(self.output_size, hidden_dim) * 0.01
		#print(self.V.shape)
		#print(V.shape)
		# bias vector of size m x 1 in equation for at
		self.b = np.zeros((hidden_dim, 1))
		# bias vector of size C x 1 in equation for ot
		self.c = np.zeros((self.output_size, 1))

		# Adagrad params

		self.ada_W = np.zeros((hidden_dim, hidden_dim))
		self.ada_U = np.zeros((hidden_dim, self.input_size))
		self.ada_V = np.zeros((self.output_size, hidden_dim))
		self.ada_b = np.zeros((hidden_dim, 1))
		self.ada_c = np.zeros((self.output_size, 1))

	def forward(self, x, y):
		# Access the previous state to calculate the current state
		h = {}
		h[-1] = np.copy(self.h)
		p = {}
		seq_length = len(x)
		loss = 0

		for t in range(seq_length):
			# One hot x
			x_t = np.zeros((self.input_size, 1))
			x_t[x[t]] = 1

			# find new hidden state
			a_t = np.dot(self.U, x_t) + np.dot(self.W, h[t-1]) + self.b
			h[t] = np.tanh(a_t)

			# unnormalized log probabilities for next chars o_t
			o_t = np.dot(self.V, h[t]) + self.c

			# Softmax
			p[t] = np.exp(o_t) / np.sum(np.exp(o_t))

			# cross-entropy loss
			loss += -np.log(p[t][y[t], 0])

		return loss, p, h

	def backward(self, x, y, p, h):
		print("in backward h:")
		print(h)
		# derivatives w.r.t different model params
		dU = np.zeros_like(self.U)
		dW = np.zeros_like(self.W)
		dV = np.zeros_like(self.V)
		db = np.zeros_like(self.b)
		dc = np.zeros_like(self.c)
		dh_next = np.zeros_like(self.h)

		for t in reversed(range(len(x))):
			# One hot y
			y_t = np.zeros((self.input_size, 1))
			y_t[y[t]] = 1

			# One hot x
			x_t = np.zeros((self.input_size, 1))
			x_t[x[t]] = 1

			# gradient w.r.t. o_t
			g = - (y_t - p[t])

			# gradient w.r.t. V and c
			dV += np.dot(g, h[t].T)
			dc += g

			# gradient w.r.t. h, tanh nonlinearity
			dh = (1 - h[t] ** 2) * (np.dot(self.V.T, g) + dh_next)

			# gradient w.r.t. U
			dU += np.dot(dh, x_t.T)

			# gradient w.r.t W
			dW += np.dot(dh, h[t - 1].T)

			# gradient w.r.t. b
			db += dh

			# Next (previous) dh
			dh_next = np.dot(self.W.T, dh)

		# clip to avoid exploding gradients
		dW = np.clip(dW, -5, 5)
		dU = np.clip(dU, -5, 5)
		dV = np.clip(dV, -5, 5)
		db = np.clip(db, -5, 5)
		dc = np.clip(dc, -5, 5)

		return dW, dU, dV, db, dc, h[-1]

	def adagrad_update(self, dW, dU, dV, db, dc):
		print "before update c:"
		print self.c
		raw_input()
		
		# Update W
		self.ada_W += dW * dW
		self.W += - self.learning_rate * dW / np.sqrt(self.ada_W + 1e-8)

		# Update U
		self.ada_U += dU * dU
		self.U += - self.learning_rate * dU / np.sqrt(self.ada_U + 1e-8)

		# Update V
		self.ada_V += dV * dV
		self.V += - self.learning_rate * dV / np.sqrt(self.ada_V + 1e-8)

		# Update c
		self.ada_c += dc * dc
		self.c += - self.learning_rate * dc / np.sqrt(self.ada_c + 1e-8)

		print "after update c:"
		print self.c
		raw_input()

		# Update b
		self.ada_b += db * db
		self.b += - self.learning_rate * db / np.sqrt(self.ada_b + 1e-8)

	def train(self, x, y):
		# Forward pass
		print x
		print len(x)
		loss, p, h = self.forward(x, y)
		#print h
		#print len(h)
		#print self.h
		#print len(self.h)
		#1/0
		# Backward pass
		dW, dU, dV, db, dc, h = self.backward(x, y, p, h)

		# Grad check
		#self.grad_check([self.W, self.U, self.V, self.b, self.c], [dW, dU, dV, db, dc], x, y)

		# Update hidden state
		self.h = h
		raw_input()

		# Adagrad update
		self.adagrad_update(dW, dU, dV, db, dc)

		return loss

	def synthesize(self, first, n):
		# sampled index t + 1th character in your sequence and will be the input vector for the next time-step of your RNN
		synth_inds = []

		# one hot x
		x_t = np.zeros((self.input_size, 1))
		x_t[first] = 1

		h = self.h
		for t in range(n):
			# Forward pass
			h = np.tanh(np.dot(self.U, x_t) + np.dot(self.W, h) + self.b)
			y = np.dot(self.V, h) + self.c
			p = np.exp(y) / np.sum(np.exp(y))

			# Generate random index
			cp = np.cumsum(p)
			a = np.random.uniform()
			cpa = cp - a
			ixs = np.where(cpa > 0)
			ii = ixs[0][0]

			# take sampled index as an input to next sampling
			x_t = np.zeros((self.input_size, 1))
			x_t[ii] = 1

			synth_inds.append(ii)

		return synth_inds

	# Gradient check, adopted from https://gist.github.com/karpathy/d4dee566867f8291f086#gistcomment-1508982
	def grad_check(self, params, grads, x, y):
		num_checks, delta = 10, 1e-4
		for param, dparam, name in zip(params, grads, ['dW', 'dU', 'dV', 'db', 'dc']):
			s0 = dparam.shape
			s1 = param.shape
			assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
			print name
			for i in xrange(num_checks):
				ri = int(uniform(0, param.size))
				old_val = param.flat[ri]
				param.flat[ri] = old_val + delta
				cg0, _, _ = self.forward(x, y)
				param.flat[ri] = old_val - delta
				cg1, _, _ = self.forward(x, y)
				param.flat[ri] = old_val
				grad_analytic = dparam.flat[ri]
				grad_numerical = (cg0 - cg1) / (2 * delta)
				rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
				if not math.isnan(rel_error):
					print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)

def main():
	data = DataLoader('goblet_book.txt')
	#print data.uniq_chars
	input_size = len(data.uniq_chars)
	output_size = len(data.uniq_chars)
	rnn = RNN(100, 0.1, input_size, output_size)

	seq_length = 25
	losses = []
	smooth_loss = -np.log(1.0/len(data.uniq_chars))*seq_length
	losses.append(smooth_loss)
	n_epochs = 10


	best_rnn = rnn
	lowest_loss = smooth_loss

	for e in xrange(n_epochs):
		for i in range(data.book_size / seq_length):
			# Input
			x = [data.char_to_ix[c] for c in data.book[i * seq_length:(i + 1) * seq_length]]
			# Output that should be predicted, i.e. next character
			y = [data.char_to_ix[c] for c in data.book[i * seq_length + 1:(i + 1) * seq_length + 1]]
			

			if i % 10000 == 0:
				synth_sample = rnn.synthesize(x[0], 200)
				txt = ''.join([data.ix_to_char[n] for n in synth_sample])
				print txt

			#raw_input()

			loss = rnn.train(x, y)
			#print loss
			smooth_loss = smooth_loss * 0.999 + loss * 0.001

			if i % 10000 == 0:
				print 'iteration %d, smooth_loss = %f' % (i, smooth_loss)
				losses.append(smooth_loss)

			if smooth_loss < lowest_loss:
				best_rnn = rnn
				lowest_loss = smooth_loss

		# reset rnn memory after each epoch
		print 'Iteration ' + str(e) + 'done..'
		rnn.h = np.zeros((100, 1))

	synth_sample = best_rnn.synthesize(x[0], 1000)
	txt = ''.join([data.ix_to_char[n] for n in synth_sample])
	print lowest_loss
	print txt


	plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
	plt.xlabel('Iterations, in thousands')
	plt.ylabel('loss')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()