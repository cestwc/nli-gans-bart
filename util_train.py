import torch

eos_token_idx = 2
pad_token_idx = 1
init_token_idx = 0

# the = 4

def fake_label(real_label):
	fake_label = (real_label.masked_fill(real_label < 2, -1) - torch.randint(1, 3, real_label.shape, device = real_label.device))
	return fake_label.masked_fill(fake_label < 0, 2)

def prob2idx(queryDistribution, eos_id = eos_token_idx, pad_id = pad_token_idx, init_id = init_token_idx):
	f_query = torch.argmax(queryDistribution, dim = 2)

	f_query.masked_fill_(f_query < 4, 4)

	eos_done = torch.zeros(len(f_query), dtype = torch.long, device = f_query.device)

	for i in range(1, f_query.shape[1] - 2):
		undone_musk = eos_done == 0
		mask = torch.logical_and(f_query[:, -i-1] != 4, torch.logical_or(f_query[:, -i] != 4, f_query[:, -i-2] != 4))
		f_query[:, -i].masked_fill_(torch.logical_and(undone_musk, mask), eos_id)
		eos_done.masked_fill_(torch.logical_and(undone_musk, mask), 1)

		undone_musk = eos_done == 0
		f_query[:, -i].masked_fill_(torch.logical_and(undone_musk, torch.logical_not(mask)), pad_id)

		if torch.min(eos_done) == 1:
			break

	f_query[:, f_query.shape[1] - 2].masked_fill_(undone_musk, eos_id)

	f_query[:, 0] = init_id

	eos_mask = f_query.eq(eos_id)

	if len(torch.unique(eos_mask.sum(1))) > 1:
		print(eos_mask)
		print(f_query)
		raise ValueError("Now the <eos> error happens here")

	return f_query


def pseudocat(tensors, num_classes, dim = 1):
	text = []
	distribution = []
	for i in range(len(tensors)):
		if len(tensors[i].shape) == 2:
			text.append(tensors[i])
			distribution.append(torch.nn.functional.one_hot(tensors[i], num_classes))
		else:
			t = tensors[i]
			init_token = torch.zeros(t.shape[0], 1, dtype = torch.long, device = t.device)
			t = torch.cat((torch.nn.functional.one_hot(init_token, num_classes), t), dim = 1)
			distribution.append(t)
			text.append(prob2idx(t))

	return torch.cat(text, dim = dim), torch.cat(distribution, dim = dim)

def noising(query):
	edit_mask = query > 3
	mutate_mask = torch.rand(query.shape[0], query.shape[1], device = query.device) > 0.8
	mask = torch.logical_and(edit_mask, mutate_mask)
	noise = torch.randint(4, torch.max(query), query.shape, device = query.device)
	return query.masked_scatter(mask, noise)

# from tutorial
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs
