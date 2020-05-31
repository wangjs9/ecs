import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pickle, math
from queue import Queue
import warnings
from transformers import BertModel, BertConfig
warnings.filterwarnings("ignore")

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'Senti'

        self.train_path = dataset + '/data/dataset_dialogs/train.txt'                                
        self.dev_path = dataset + '/data/dataset_dialogs/dev.txt'                                    
        self.test_path = dataset + '/data/dataset_dialogs/dev.txt'                         
                          
        self.stopwords_path = dataset + '/data/dataset_dialogs/stopwords.txt'
        self.vocab_path = dataset + '/data/dataset_dialogs/vocab.pkl'
        self.topic_path = dataset + '/data/dataset_dialogs/topic'
        self.emotion_path = dataset + '/data/dataset_dialogs/emotion.pkl'

        self.save_path = dataset + '/saved_dict/dialogs/{0}/{0}.pth'.format(self.model_name)        
        self.save_dic = dataset + '/saved_dict/dialogs/{}/'.format(self.model_name)    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.require_improvement = 10                              
        self.num_epochs = 1                                         
        self.learning_rate = 5e-4
        self.batch_size = 68                               
        self.pad_size = 30 
        self.topic_num = 0                                         

        self.dropout = 0
        self.embed_size = 512
        self.enc_layer = 1
        self.dec_layer = 1
        self.enc_hidden_size = 256
        self.dec_hidden_size = 256

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, enc_layer, embed, dropout=0.5):
        super(Encoder, self).__init__()
        self.embed = embed
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, src, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        src_sorted = src[sorted_idx.long()]
        embedded = self.dropout(self.embed(src_sorted))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        
        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)
        
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                
            batch_size, input_len, -1) # batch_size, context_len, dec_hidden_size
        
        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len 
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1,2)) 
        # batch_size, output_len, context_len
        attn.data.masked_fill(mask, -1e6)
        attn = F.softmax(attn, dim=2) # batch_size, output_len, context_len
        context = torch.bmm(attn, context) # batch_size, output_len, enc_hidden_size
        
        output = torch.cat((context, output), dim=2) # batch_size, output_len, hidden_size*2

        output = output.view(batch_size*output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dec_layer, embed, dropout=0.5):
        super(Decoder, self).__init__()
        self.layer = dec_layer
        self.embed = embed
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.emotion_out = External_Memory(dec_hidden_size)
        self.grammar_out = Filter(dec_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = (torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]).float()
        y_mask = (torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]).float()
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask
    
    def forward(self, ctx, ctx_lengths, tgt, tgt_lengths, hid, tgt_label, emotion, grammar):
        sorted_len, sorted_idx = tgt_lengths.sort(0, descending=True)
        y_sorted = tgt[sorted_idx.long()] # batch_size, output_length, embed_size
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))
        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        
        out, hid = self.rnn(packed_seq, hid.repeat(self.layer, 1, 1))
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(tgt_lengths, ctx_lengths)

        output, attn = self.attention(output_seq, ctx, mask)
        output = self.out(output)
        output = self.emotion_out(hid, output, tgt_label, emotion)
        output = self.grammar_out(output, hid, grammar)
        output = F.log_softmax(output, -1)
        
        return output, hid, attn

class External_Memory(nn.Module):
    def __init__(self, dec_hidden_size):
        super(External_Memory, self).__init__()
        self.arfa = nn.Linear(dec_hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, state, output, tgt_label, emotion):
        arfa = self.sigmoid(self.arfa(state.transpose(0,1)))
        memory = emotion[tgt_label].unsqueeze(1).float()
        generic = torch.ones_like(memory) - memory
        emotion_out = torch.mul(arfa, torch.mul(output, memory))
        generic_out = torch.mul((1-arfa), torch.mul(output, generic))

        output = torch.add(emotion_out, generic_out)
        return output

class Filter(nn.Module):
    def __init__(self, dec_hidden_size):
        super(Filter, self).__init__()
        self.arfa = nn.Linear(dec_hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, output, state, grammar):
        arfa = self.sigmoid(self.arfa(state.transpose(0,1)))

        memory = torch.zeros(output.size(-1), device=output.device)
        memory[grammar] = 1
        memory = memory.unsqueeze(0).float().repeat(output.size(0),1).unsqueeze(1)

        generic = torch.ones_like(memory) - memory
        grammar_out = torch.mul(arfa, torch.mul(output, memory))
        generic_out = torch.mul(output, generic)

        output = torch.add(grammar_out, generic_out)
        return output

class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__()
        self.vocab_size = len(pickle.load(open(config.vocab_path, 'rb')))
        embed = nn.Embedding(self.vocab_size, config.embed_size)
        self.encoder = Encoder(self.vocab_size, config.embed_size, config.enc_hidden_size, config.dec_hidden_size, config.enc_layer, embed, config.dropout)
        self.decoder = Decoder(self.vocab_size, config.embed_size, config.enc_hidden_size, config.dec_hidden_size, config.dec_layer, embed, config.dropout)
        self.pad_size = config.pad_size
        self.device = config.device
        
    def forward(self, Queries, Responses, emotion, grammar=None):
        src = Queries[0]
        tgt = Responses[0]
        src_lengths = Queries[1]
        tgt_lengths = Responses[1]
        tgt_label = Responses[2]
        batch_size = src.shape[0]
        tgt = torch.cat((torch.ones(batch_size, 1, device=src.device, dtype=torch.long) * 1, tgt), dim=-1)[:, :self.pad_size]
        
        encoder_out, hid = self.encoder(src, src_lengths)
        output, hid, attn = self.decoder(encoder_out, src_lengths,
                    tgt, tgt_lengths, hid, tgt_label, emotion, grammar)
        add = torch.zeros(output.shape[0], self.pad_size-output.shape[1], output.shape[2]).to(self.device)
        out = torch.cat((output, add), dim = 1)
        return out, attn
    
    def response(self, Queries, tgt_labels, emotion=None, grammar=None, max_length=32, beam_width=2):
        src = Queries[0]
        src_lengths = Queries[1]
        batch_size = src.shape[0]
        encoder_outs, hids = self.encoder(src, src_lengths)
        decoded_batch = []
        
        for idx in range(batch_size):
            encoder_out = encoder_outs[idx, :, :].unsqueeze(0)
            hid = hids[:, idx, :].unsqueeze(1)
            tgt_label = tgt_labels[idx].unsqueeze(0)

            tgt = torch.Tensor([[1]]).long().to(self.device)
            
            root = BeamSearchNode(hid, None, tgt, 0, 1)
            q = Queue()
            q.put(root)
            end_nodes = []
            
            while not q.empty():
                candidates = []
                cur_length = q.qsize()
                for _ in range(cur_length):
                    node = q.get()
                    tgt = node.wordId
                    hid = node.hid
                    cur_length = node.length

                    if tgt.item() == 2 or node.length >= max_length:
                        end_nodes.append((node.score(), node))
                        continue

                    output, hid, attn = self.decoder(encoder_out, src_lengths, tgt,
                            torch.ones(1).long().to(tgt.device),
                            hid, tgt_label, emotion, grammar)
                    log_prob, indexes = torch.topk(output.squeeze(0), dim=-1, k=beam_width)
                    
                    for k in range(beam_width):
                        index = indexes[0][k].view(1, -1)
                        log_p = log_prob[0][k]
                        child = BeamSearchNode(hid, node, index, node.logp+log_p, node.length+1)
                        candidates.append((child.score(), child))
                        
                candidates = sorted(candidates, key=lambda  x:x[0], reverse=True)
                length = min(len(candidates), beam_width)
                for i in range(length):
                    q.put(candidates[i][1])
            
            if len(end_nodes) == 0:
                node = q.get()
            else:
                node = sorted(end_nodes,  key=lambda  x:x[0], reverse=True)[0][1]
            utterance = [node.wordId.item()]
            while node.prevNode != None:
                node = node.prevNode
                utterance.append(node.wordId.item())
            utterance = utterance[-2::-1]
            decoded_batch.append(utterance)

        return decoded_batch  
        
    def greedy_response(self, Queries, tgt_label, emotion=None, grammar=None, max_length=32):
        src = Queries[0]
        src_lengths = Queries[1]
       
        batch_size = src.shape[0]
        preds = []
        
        encoder_out, hid = self.encoder(src, src_lengths)
        tgt = torch.Tensor([[1] for i in range(batch_size)]).long().to(self.device)
        for i in range(max_length):
            output, hid, attn = self.decoder(encoder_out, src_lengths, tgt,
                    torch.ones(batch_size).long().to(tgt.device),
                    hid, tgt_label, emotion, grammar)

            tgt = output.max(2)[1].view(batch_size, 1)
            preds.append(tgt)
            
        return torch.cat(preds, 1).cpu().tolist()

class BeamSearchNode(object):
    def __init__(self, hid, previousNode, wordId, logProb, length):
        self.hid = hid
        self.prevNode = previousNode
        self.wordId = wordId
        self.logp = logProb
        self.length = length
    
    def score(self, alpha=1):
        return self.logp / self.length
