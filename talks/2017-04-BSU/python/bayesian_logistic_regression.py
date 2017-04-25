# MODEL
x = tf.placeholder(tf.float32, [N, D])
beta = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
y = Bernoulli(logits=ed.dot(x, beta))

# INFERENCE
qbeta = Empirical(params=tf.Variable(tf.random_normal([T, D])))
inference = ed.HMC({beta: qbeta}, data={y: y_data})
