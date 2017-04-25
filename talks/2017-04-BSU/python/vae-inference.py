inference = ed.KLqp({z: qz}, data={x: qx})
inference.initialize()

for epoch in range(n_epoch):
  for t in range(n_iter_per_epoch):
    inference.update(feed_dict={x_ph: mnist.train.next_batch(M)})
