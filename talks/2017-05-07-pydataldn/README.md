## Bayesian Deep Learning with Edward (and a trick using Dropout)

[Andrew Rowan](https://twitter.com/arowan_ml)

[PyData London 2017](https://pydata.org/london2017/)

Sunday, 7 May, 11am

* Slides: <http://rpubs.com/arowan/bayesian_deep_learning>

* Recording: <https://www.youtube.com/watch?v=I09QVNrUS3Q&t=1170s>

[Abstract](<https://pydata.org/london2017/schedule/presentation/33/>)

Deep learning methods represent the state-of-the-art for many applications such as speech recognition, computer vision and natural language processing.  Conventional approaches generate point estimates of deep neural network weights and hence make predictions that can be overconfident since they do not account well for uncertainty in model parameters.  However, having some means of quantifying the uncertainty of our predictions is often a critical requirement in fields such as medicine, engineering and finance.  One natural response is to consider Bayesian methods, which offer a principled way of estimating predictive uncertainty while also showing robustness to overfitting.  

Bayesian neural networks have a long history.  Exact Bayesian inference on network weights is generally intractable and much work in the 1990s focused on variational and Monte Carlo based approximations [1-3].  However, these suffered from a lack of scalability for modern applications.  Recently the field has seen a resurgence of interest, with the aim of constructing practical, scalable techniques for approximate Bayesian inference on more complex models, deep architectures and larger data sets [4-10].

Edward is a new, Turing-complete probabilistic programming language built on Python [11].  Probabilistic programming frameworks typically face a trade-off between the range of models that can be expressed and the efficiency of inference engines.  Edward can leverage graph frameworks such as TensorFlow to enable fast distributed training, parallelism, vectorisation, and GPU support, while also allowing composition of both models and inference methods for a greater degree of flexibility. 

In this talk I will give a brief overview of developments in Bayesian deep learning and demonstrate results of Bayesian inference on deep architectures implemented in Edward for a range of publicly available data sets.  Dropout is an empirical technique which has been very successfully applied to reduce overfitting in deep learning models [12].  Recent work by Gal and Ghahramani [13, 14] has demonstrated a surprising formal equivalence between dropout and approximate Bayesian inference in neural networks.  I will compare some results of inference via the machinery of Edward with model averaging over neural nets with dropout training. 

**References**

1.	D JC MacKay. *A Practical Bayesian Framework for Backpropagation Networks*, Neural Computation, 4(3): 448–472, (1992). 
2.	R Neal. *Bayesian Learning for Neural Networks*, PhD thesis, University of Toronto, (1995). 
3.	G Hinton, D Van Camp. *Keeping Neural Networks Simple by Minimizing the Description Length of the Weights*, Proceedings of the 6th annual conference on computational learning theory, (1993). 
4.	A Graves. *Practical Variational Inference for Neural Networks*, NIPS 2011. 
5.	D Kingma, T Salimans, M Welling, *Variational Dropout and the Local Reparameterization Trick*, https://arxiv.org/pdf/1506.02557 (2015) 
6.	A Mnih, K Gregor, *Neural Variational Inference and Learning in Belief Networks*, ICML 2014 
7.	D Kingma, M Welling, *Auto-Encoding Variational Bayes*, CoRR abs/1312.6114 (2013) 
8.	D Rezende, S Mohamed, D Wierstra. *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*, ICML 2014. 
9.	C Blundell, J Cornebise, K Kavukcuoglu, D Wierstra, *Weight Uncertainty in Neural Networks*, ICML 2015. 
10.	J M Hernandez-Lobato, R P Adams, *Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks*, ICML 2015
11.	D Tran, A Kucukelbir, A B Dieng, M Rudolph, D Liang, D M Blei, *Edward: a Library for Probabilistic Modeling, Inference, and Criticism*, arXiv:1610.09787 (2016) 
12.	N Srivastava, G Hinton, A Krizhevsky, I Sutskever, R Salakhutdinov, *Dropout: a Simple Way to Prevent Neural Networks from Overfitting*, Journal of Machine Learning Research, 15(1), (2014). 
13.	Y Gal, Z Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML 2016
14. Y Gal, *Uncertainty in Deep Learning*, PhD Thesis, University of Cambridge (2016)
