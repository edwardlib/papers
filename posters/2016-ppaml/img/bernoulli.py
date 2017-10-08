import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational, Dirichlet, Normal, InvGamma
from edward.stats import dirichlet, invgamma, multivariate_normal, norm
from edward.util import get_dims

class MixtureGaussian:

    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.num_vars = (2*D + 1) * K

        self.a = 1
        self.b = 1
        self.c = 10
        self.alpha = tf.ones([K])

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        N = get_dims(xs)[0]
        pi, mus, sigmas = zs
        log_prior = dirichlet.logpdf(pi, self.alpha)
        log_prior += tf.reduce_sum(norm.logpdf(mus, 0, np.sqrt(self.c)), 1)
        log_prior += tf.reduce_sum(invgamma.logpdf(sigmas, self.a, self.b), 1)

        # Loop over each mini-batch zs[b,:]
        log_lik = []
        n_minibatch = get_dims(zs[0])[0]
        for s in range(n_minibatch):
            log_lik_z = N*tf.reduce_sum(tf.log(pi), 1)
            for k in range(self.K):
                log_lik_z += tf.reduce_sum(multivariate_normal.logpdf(xs,
                    mus[s, (k*self.D):((k+1)*self.D)],
                    sigmas[s, (k*self.D):((k+1)*self.D)]))

            log_lik += [log_lik_z]

        return log_prior + tf.pack(log_lik)

ed.set_seed(42)
x = np.loadtxt('data/mixture_data.txt', dtype='float32', delimiter=',')
data = ed.Data(tf.constant(x, dtype=tf.float32))

model = MixtureGaussian(K=2, D=2)
variational = Variational()
variational.add(Dirichlet(model.K))
variational.add(Normal(model.K*model.D))
variational.add(InvGamma(model.K*model.D))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=500, n_minibatch=5, n_data=5)
