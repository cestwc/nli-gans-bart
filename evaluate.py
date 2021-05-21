import torch
from tqdm.notebook import tqdm

from util_train import categorical_accuracy

def evaluate(model, iterator, criterion):

	epoch_loss = 0
	epoch_acc = 0

	model.eval()

	with torch.no_grad():

		for _, batch in enumerate(tqdm(iterator)):

			delimiter = torch.ones(batch.query.shape[0], 1, device = batch.query.device, dtype = torch.long) * 2
			predictions = model(torch.cat((batch.text, delimiter, batch.query), dim = 1)).logits

			loss = criterion(predictions, batch.label)

			acc = categorical_accuracy(predictions, batch.label)

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)
