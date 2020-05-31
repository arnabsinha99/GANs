Model structure:

Encoder(
  (hidden1): Sequential(
    (0): Linear(in_features=2, out_features=10, bias=True)
    (1): LeakyReLU(0.2)
  )
  (hidden2): Sequential(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): LeakyReLU(0.2)
  )
  (out): Sequential(
    (0): Linear(in_features=10, out_features=2, bias=True)
    (1): LeakyReLU(0.2)
  )
)

Results: Single and Double modality data learnt well with error close to 0.009.