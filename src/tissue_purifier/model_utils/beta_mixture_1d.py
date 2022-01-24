import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

# This block of code is taken from:
# https://github.com/PaulAlbert31/LabelNoiseCorrection
# It fits the CrossEntropyLoss of a classifier with a two components Beta Distribution using the EM algorithm.
# Low losses correspond to correct labels, High losses to incorrect labels.
# For each instance you get the assignment probability, w_i, to the correct and incorrect component. 
# The number of classes in the classifier can be LARGER than 2. 
# The corrected labels are computed as:
# new_label = (1-w_i) * y_i + w_i * z_i
# where:
# a. y_i is the (possibly incorrect) hard label
# b. z_i is the label generated by the classifier (can be either one-hot or soft)
# c. w_i is the probability of the label being incorrect


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(
            self, 
            max_iters=10,
            alphas_init=[2, 5],
            betas_init=[5, 2],
            weights_init=[0.5, 0.5]):
        # save the initial values
        self.alphas_init = alphas_init
        self.betas_init = betas_init
        self.weights_init = weights_init
        # other parameters
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12
        # for debug
        self.empirical_loss = None

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def reset_param(self):
        self.alphas = np.array(self.alphas_init, dtype=np.float64)
        self.betas = np.array(self.betas_init, dtype=np.float64)
        self.weight = np.array(self.weights_init, dtype=np.float64)

    def switch_param(self):
        tmp_alphas = np.copy(self.alphas[::-1])
        tmp_betas = np.copy(self.betas[::-1])
        tmp_weight = np.copy(self.weight[::-1])
        self.alphas = tmp_alphas
        self.betas = tmp_betas
        self.weight = tmp_weight

    def fit(self, x, reset: bool = False):
        # set the parameters to their initial value to start the EM algorithm
        if reset:
            self.reset_param()

        x = np.copy(x)
        self.empirical_loss = x  # store the value which have been used to fit the model

        # EM on beta distributions un-usable with x == 0 or x==1
        eps = 1e-4
        x = np.clip(x, a_min=eps, a_max=1-eps)

        # EM algorithm
        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            # on rare occasion the two components can be switched
            # (i.e. 0 -> high value and 1 -> low value).
            # If this is the case, switch them back
            fitted_means_tmp = self.alphas / (self.alphas + self.betas)
            if fitted_means_tmp[0] > fitted_means_tmp[1]:
                self.switch_param()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.hist(self.empirical_loss, bins=50, density=True)
        plt.legend()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
