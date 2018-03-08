import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

# training
def train(decoder, optimizer, criterion, len_targets, all_characters, batch_size, inp, target, attn=False, ponder=False):
    loss = 0
    hidden = decoder.init_hidden(batch_size)
    decoder.zero_grad()
    teacher_forcing_ratio = 0.5
    if attn:
        output, ht, hidden, vis_attn = decoder(inp, hidden, batch_size)
        grammar_contexts = ht
        ht = decoder.init_prev_ht(batch_size, 1)
        # choose whether to use teacher forcing #
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        # use_teacher_forcing = False
        dec_input = Variable(torch.ones(batch_size).long() * all_characters.index("SOS")).cuda()
        for c in range(len_targets):
            if ponder:
                output, ht, hidden, vis_attn = decoder(target[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                                       context=grammar_contexts, prev_ht=ht, ponder=True)
            else:
                if use_teacher_forcing:
                    output, ht, hidden, vis_attn = decoder(dec_input.unsqueeze(1), hidden, batch_size, attn=True,
                                                           context=grammar_contexts, prev_ht=ht, ponder=False)
                    loss += criterion(output.view(batch_size, -1), target[:, c])
                    dec_input = target[:, c]
                else:
                    output, ht, hidden, vis_attn = decoder(dec_input.unsqueeze(1), hidden, batch_size, attn=True,
                                                           context=grammar_contexts, prev_ht=ht, ponder=False)
                    loss += criterion(output.view(batch_size, -1), target[:, c])
                    # Get most likely word index (highest value) from output
                    topv, topi = output.data.topk(1)
                    top_pred = Variable(topi.squeeze(1)).cuda()
                    # test = Variable(torch.ones(batch_size).long() * topi).cuda()

                    dec_input = top_pred
                    # print(target[:,c], output.max(1)[1])
    else:
        output, ht, hidden, _ = decoder(inp, hidden, batch_size)
        # choose whether to use teacher forcing #
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        # use_teacher_forcing = False
        dec_input = Variable(torch.ones(batch_size).long() * all_characters.index("SOS")).cuda()
        for c in range(len_targets):
            if use_teacher_forcing:
                output, ht, hidden, _ = decoder(dec_input.unsqueeze(1), hidden, batch_size)
                loss += criterion(output.view(batch_size, -1), target[:, c])
                dec_input = target[:, c]
                # print(target[:,c], output.max(1)[1])
            else:
                output, ht, hidden, _ = decoder(dec_input.unsqueeze(1), hidden, batch_size)
                loss += criterion(output.view(batch_size, -1), target[:, c])
                # Get most likely word index (highest value) from output
                topv, topi = output.data.topk(1)
                top_pred = Variable(topi.squeeze(1)).cuda()
                # test = Variable(torch.ones(batch_size).long() * topi).cuda()

                dec_input = top_pred
                # print(target[:,c], output.max(1)[1])

    loss.backward()
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)
    optimizer.step()

    return loss.data[0] / len_targets