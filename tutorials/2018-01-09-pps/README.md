# slides: https://braintex.goog/project/5a4da80b74bc09007bb19055
# poster: https://braintex.goog/project/5a4c6fe174bc09007bb18ffc

https://conf.researchr.org/track/POPL-2018/pps-2018

Title:
Deep Probabilistic Programming: TensorFlow Distributions and Edward

Abstract:
The TensorFlow Distribution and Edward libraries implement a vision of
probability theory adapted to the modern deep-learning paradigm of
end-to-end differentiable computation. We first introduce TensorFlow
Distributions, an efficient low-level system for building and
manipulating distributions. We focus on the non-obvious design choices
in the library, paying particular attention to the Bijector
abstraction, which supports composable volume-tracking transformations
with automatic caching. We then provide an overview of Edward, a
probabilistic programming system built on computational graphs and
using Distributions as an efficient backend. In particular, we show
how Edward and TensorFlow Distributions can be applied for expanding
the frontier of deep generative models and variational inference.
