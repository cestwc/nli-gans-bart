# nli-gans-bart
use BART in both the generator and discriminator

## Usage
We take the nli version of FEVER as an example, though the GANs here may not necessarily boost the performance, and will slow the training down by half.

Prepare data
```python
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.max_model_input_sizes['facebook/bart-base']

def tokenize_and_cut(sentence):
	tokens = tokenizer.tokenize(sentence) 
	tokens = tokens[:max_input_length-2 - 113] # we need to concatenate the context and query
	return tokens

from torchtext.legacy import data

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

QUERY = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField()

from torchtext.legacy import data

fields = {'context': ('text', TEXT), 'query': ('query', QUERY), 'label': ('label', LABEL)}

all_data, = data.TabularDataset.splits(
                            path = '.',
                            train = your_file_name,
                            format = 'json',
                            fields = fields
)

train_data, test_data = all_data.split(random_state = random.seed(SEED))
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)

BATCH_SIZE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x : len(x.text),
    shuffle = True,
    device = device)
```

Build the Model
```python
import torch
import torch.nn as nn

from transformers import BartForConditionalGeneration
from extended_model import BartForSequenceClassificationWithSoftInputs

netD = BartForSequenceClassificationWithSoftInputs.from_pretrained('facebook/bart-base')
netG = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The discriminator has {count_parameters(netD):,} trainable parameters')
print(f'The generator has {count_parameters(netG):,} trainable parameters')
```

Train the model
```python
LEARNING_RATE = 2e-5#0.0005
beta1 = 0.5

optimizerD = torch.optim.Adam(netD.parameters(), lr = LEARNING_RATE, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr = LEARNING_RATE / 4, betas=(beta1, 0.999))

criterionD = nn.CrossEntropyLoss()
criterionG = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

netD = netD.to(device)
netG = netG.to(device)
criterionD = criterionD.to(device)
criterionG = criterionG.to(device)

from train import train
from evaluate import evaluate

train_lossG, train_lossD, train_acc = train(netD, netG, train_iterator, optimizerD, optimizerG, criterionD, criterionG, valid_iterator, N_EPOCHS = 2, dirD = drivePath + 'netD.pt', dirG = drivePath + 'netG.pt', interval = 50, valid_interval = 1000)


```

Test
```python
netD.load_state_dict(torch.load(drivePath + 'netD.pt', map_location = device))

test_loss, test_acc = evaluate(netD, test_iterator, criterionD)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```
```
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(train_lossG,label="G")
plt.plot(train_lossD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

Inference
```python
def predict_sentiment(model, tokenizer, sentence):
	model.eval()
	tokens = tokenizer.tokenize(sentence)
	tokens = tokens[:max_input_length-2]
	indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
	tensor = torch.LongTensor(indexed).to(device)
	tensor = tensor.unsqueeze(0)
	prediction = model(tensor).logits
	return prediction
```
```
predict_sentiment(netD, tokenizer, "This film looks great. </s> The film is terrible")
```
