import numpy as np
import matplotlib.pyplot as plt


class Bandit:
	def __init__(self,true_mean):
		self.true_mean = true_mean
		self.estimate_mean = 0
		self.N = 0

	def pull(self):
		"""
		** Evry Bandit reward will be a Gaussian with unit of variance/variants
		randn() A single float randomly sampled from the distribution is returned
		"""
		return np.random.randn() + self.true_mean

	def update(self, x):
		self.N +=1
		self.estimate_mean = (1-1.0/self.N) * self.estimate_mean + 1.0/self.N * x

def run_experiment(m1,m2,m3, epsilon, N):
	"""
	:param m1: TrueMean for 1st
	:param m2: ""
	:param m3: ""
	:param epsilon: value to be used for epsilon greedy
	:param N: number of times to play
	:return: an array containing the cumulative average after every plaot
	"""
	bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

	# Initialise an empty array with Shape N
	data = np.empty(N)

	# Our goal is here is to
	for i in range(N):
		#epsilon greedy
		p = np.random.random() # generate a random number between 0-1
		if p < epsilon:        # We explore/choose a random bandit
			n_bandit = np.random.choice(3)
		else:                  # We choose the bandit with the best estimate_mean
			n_bandit =  np.argmax([b.estimate_mean for b in bandits])


		x = bandits[n_bandit].pull()
		bandits[n_bandit].update(x)

		data[i] = x

	cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

	#plot moving average ctr

	plt.plot(cumulative_average)
	plt.plot(np.ones(N) * m1)
	plt.plot(np.ones(N) * m2)
	plt.plot(np.ones(N) * m3)
	plt.xscale('log')
	plt.show()

	for b in bandits:
		print(b.estimate_mean)

	return  cumulative_average





if __name__ == '__main__':
	eps_15 = run_experiment(1.0,3.0,2.0, epsilon=0.15, N=1000)
	# eps_10 = run_experiment(1.0,3.0,2.0, epsilon=0.1, N=1000)
	# eps_05 = run_experiment(1.0,3.0,2.0, epsilon=0.05, N=1000)
	# eps_01 = run_experiment(1.0,3.0,2.0, epsilon=0.01, N=1000)

	# Log Plot
	# plt.plot(eps_15, label='eps = 0.15')
	# plt.plot(eps_10, label='eps = 0.1')
	# plt.plot(eps_05, label='eps = 0.05')
	# plt.plot(eps_01, label='eps = 0.01')
	# plt.legend()
	# plt.xscale('log')
	# plt.show()

	# Linear Plot
	# plt.plot(eps_15, label='eps = 0.15')
	# plt.plot(eps_10, label='eps = 0.1')
	# plt.plot(eps_05, label='eps = 0.05')
	# plt.plot(eps_01, label='eps = 0.01')
	# plt.legend()
	# plt.show()


# plt.hist(np.random.randn(1000), bins = 100)

