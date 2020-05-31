Model structure:

Encoder(
  (hidden1): Sequential(
    (0): Linear(in_features=2, out_features=10, bias=True)
    (1): Sigmoid()
  )
  (hidden2): Sequential(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): Sigmoid()
  )
  (out): Sequential(
    (0): Linear(in_features=10, out_features=2, bias=True)
    (1): Sigmoid()
  )
)

Results: 1. Single and Double modality data learnt well with error close to 0.007.
	 2. Output better than sigmoid activation in hidden layer. Autoencoder trying to predict values
	    closer to 0 and 1.	