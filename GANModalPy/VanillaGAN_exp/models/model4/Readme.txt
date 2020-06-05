Corrections:

1. train_d() function now performs only one backward pass for both errors.
2. Main training loop uses 1 noise for both model's training
3. Batch size increased to 100
4. Learning rate was first 0.01 till 1000 epochs. Then changed to 0.0002 after that.

Result: 1. correct minmax graph for generator and discriminator. 
	2. correct distribution graph


