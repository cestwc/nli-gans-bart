import torch
import numpy as np
from tqdm.notebook import tqdm

from util_train import categorical_accuracy, fake_label, pseudocat, noising

def train(d, g, iterator, optD, optG, lossD, lossG):

	# Training Loop

	# Lists to keep track of progress
	G_losses = []
	D_losses = []
	D_acc = []

	d.train()
	g.train()

	for i, batch in enumerate(tqdm(iterator)):

		text = batch.text
		query = batch.query

		optG.zero_grad()
		optD.zero_grad()

		############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		## Train with all-real batch        

		delimiter = torch.ones(query.shape[0], 1, device = query.device, dtype = torch.long) * 2

		# Forward pass real batch through D
		predictions = d(input_ids = torch.cat((text, delimiter, query), dim = 1)).logits

		# Calculate loss on all-real batch
		errD_real = lossD(predictions, batch.label)

		acc = categorical_accuracy(predictions, batch.label)

		# Calculate gradients for D in backward pass        
		errD_real.backward()


		## Train with all-fake batch
		# Generate batch of latent vectors
		noise = torch.randint(11, 50000, (query.shape[0], 10), device = query.device, dtype = torch.long)

		# Generate fake query batch with G
		f_query = g(torch.cat((noise, text), dim = 1), decoder_input_ids = query[:,:-1])[0]

		#fake = [batch size, query len - 1, output dim]
		#query = [batch size, query len]

		f_label = fake_label(batch.label)

		# Classify all fake batch with D
		f_cat_text, f_cat_embed = pseudocat([text, delimiter, f_query.detach()], f_query.shape[-1], dim = 1)

		f_predictions = d(input_ids = f_cat_text, inputs_probs = f_cat_embed).logits

		# Calculate D's loss on the all-fake batch
		errD_fake = lossD(f_predictions, f_label)

		# Calculate the gradients for this batch, accumulated (summed) with previous gradients
		errD_fake.backward()

		# Compute error of D as sum over the fake and the real batches
		errD = errD_real + errD_fake

		# Update D
		optD.step()


		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################

		# Since we just updated D, perform another forward pass of all-fake batch through D
		f_cat_text, f_cat_embed = pseudocat([text, delimiter, f_query], f_query.shape[-1], dim = 1)
		f_predictions = d(input_ids = f_cat_text, inputs_probs = f_cat_embed).logits

		f_query = f_query.contiguous().view(-1, f_query.shape[-1])
		n_query = noising(query)[:,1:].contiguous().view(-1)

		#f_query = [batch size * trg len - 1, output dim]
		#n_query = [batch size * trg len - 1]

		# Calculate G's loss based on this output
		errG = lossD(f_predictions, batch.label) + lossG(f_query, n_query)

		# Calculate gradients for G
		errG.backward()

		# Update G
		optG.step()

		# Output training stats
		if i % 5000 == 0:
			print('[%d/%d]\tLoss_D: %.2f\tLoss_G: %.2f\tAcc: %.2f'
				  % (i, len(iterator),
					 errD.item(), errG.item(), acc.item()))

		# Save Losses for plotting later
		G_losses.append(errG.item())
		D_losses.append(errD.item())
		D_acc.append(acc.item())

	return np.mean(G_losses), np.mean(D_losses), np.mean(D_acc)
