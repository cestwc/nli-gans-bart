import time
import torch
import numpy as np
from tqdm.notebook import tqdm

from util_train import categorical_accuracy, fake_label, pseudocat, noising, epoch_time
from evaluate import evaluate

def train(d, g, iterator, optD, optG, lossD, lossG, valid_iterator, N_EPOCHS = 3, dirD = 'netD.pt', dirG = 'netG.pt', best_valid_loss = float('inf'), interval = 5, valid_interval = 50):

	# Training Loop

	# Lists to keep track of progress
	G_losses = []
	D_losses = []
	D_acc = []
	iters = 0

	d.train()
	g.train()

	# For each epoch
	for epoch in range(N_EPOCHS):
		
		start_time = time.time()
	
		# For each batch in the iterator
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
			if i % interval == 0:
				print(f'[{i:d}/{len(iterator):d}]\tTrain G Loss: {errG.item():.2f} | Train D Loss: {errD.item():.2f} | Train Acc: {acc.item()*100:.1f}%')
				
			iters += 1
			
			if (iters % valid_interval == 0) or ((epoch == N_EPOCHS-1) and (i == len(iterator)-1)):
				valid_loss, valid_acc = evaluate(d, valid_iterator, lossD)
				if valid_loss < best_valid_loss:
					best_valid_loss = valid_loss
					torch.save(d.state_dict(), dirD)
					torch.save(g.state_dict(), dirG)
				print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
				d.train()				

			# Save Losses for plotting later
			G_losses.append(errG.item())
			D_losses.append(errD.item())
			D_acc.append(acc.item())
		
		end_time = time.time()
		
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		
		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


	return G_losses, D_losses, D_acc
