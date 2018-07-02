import torch.nn.functional as F
import torch
from torch.autograd import Variable


def train(inp, target, rnn, rnn_optimizer, criterion, batch_size, len_example, n_hidden, attn=False, ponder=False, pool=False):
    hidden = rnn.init_hidden(batch_size)
    rnn.train()
    rnn.zero_grad()
    # pool
    pool_hidden = Variable(torch.zeros(len_example - 20, batch_size, 1
                                           , n_hidden)).cuda()
    if attn:
        output, ht, hidden, vis_attn = rnn(inp[:,:20], hidden, batch_size)
        grammar_contexts = ht
        ht = rnn.init_prev_ht(batch_size, 1)
        for c in range(len_example - 20):
            if ponder:
                output, ht, hidden, vis_attn = rnn(inp[:,20+c].unsqueeze(1), hidden, batch_size, attn=True,
                                     context=grammar_contexts, prev_ht=ht, ponder=True)
            else:
                output, ht, hidden, vis_attn = rnn(inp[:,20+c].unsqueeze(1), hidden, batch_size, attn=True,
                                     context=grammar_contexts, prev_ht=ht, ponder=False)
            pool_hidden[c] = ht
        # pool
        if pool:
            logits, _ = torch.max(pool_hidden, 0)
            cls = rnn.output(logits.squeeze(1))
        else:
            cls = output
        cls = F.softmax(cls, dim=1)
    else:
        output, ht, hidden, vis_attn = rnn(inp, hidden, batch_size)
        cls = output[:,-1,:].contiguous()
        cls = F.softmax(cls, dim=1)
    loss = criterion(cls.view(batch_size, -1), target.squeeze(1))
    loss.backward()
    rnn_optimizer.step()

    return loss.data[0], ht, hidden