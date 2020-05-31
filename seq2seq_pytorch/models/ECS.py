import torch 
import torch.nn as nn
import torch.nn.functional as F 
from queue import Queue
import numpy as np 
import pickle, math, time
import warnings
from transformers import BertModel, BertConfig
warnings.filterwarnings("ignore")

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'ECS'

        self.train_path = dataset + '/data/dataset_dialogs/train.txt'                                
        self.dev_path = dataset + '/data/dataset_dialogs/dev.txt'                                    
        self.test_path = dataset + '/data/dataset_dialogs/test.txt'                                  

        self.stopwords_path = dataset + '/data/dataset_dialogs/stopwords.txt'
        self.vocab_path = dataset + '/data/dataset_dialogs/vocab.pkl'
        self.emotion_path = dataset + '/data/dataset_dialogs/emotion.pkl'
        self.topic_path = dataset + '/data/dataset_dialogs/topic'
        
        self.save_path = dataset + '/saved_dict/dialogs/{0}/{0}.pth'.format(self.model_name)        
        self.save_dic = dataset + '/saved_dict/dialogs/{}/'.format(self.model_name)    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.require_improvement = 15
        self.num_epochs = 1         
        self.learning_rate = 5e-4
        self.batch_size = 60
        self.pad_size = 30  
        self.topic_num = 20                                  

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
        self.rnn = nn.GRU(embed_size, enc_hidden_size, enc_layer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.linear_normal = nn.Linear(enc_hidden_size*2, embed_size, bias=False)

    def forward(self, src, lengths, topics):
        batch_size = src.size(0)
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

        topic_embed = torch.cat((self.linear_normal(hid).view(batch_size, 1, -1), self.dropout(self.embed(topics))), dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        
        return out, hid, topic_embed

class TopicAttn(nn.Module):
    def __init__(self, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.5):
        super(TopicAttn, self).__init__()
        self.linear_in = nn.Linear(embed_size, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(embed_size + dec_hidden_size, dec_hidden_size)
        # self.combine = nn.Linear(dec_hidden_size*2 + enc_hidden_size*2, dec_hidden_size, bias=False)

    def forward(self, output, topic_embed, topic_mask):
        # output: batch_size, output_len, dec_hidden_size
        # topic_embed: batch_size, K, embeded_size
        # hid: batch_size, enc_hidden_size * 2
        batch_size = output.size(0)
        output_len = output.size(1)
        topic_len = topic_embed.size(1)

        topics_in = self.linear_in(topic_embed.view(batch_size*topic_len, -1)).view(batch_size, topic_len, -1)
        topicattn = torch.bmm(output, topics_in.transpose(1, 2)) # batch_size, output_len, topic_len
        topicattn.data.masked_fill(topic_mask, -1e6)
        topicattn = F.softmax(topicattn, dim=2) # batch_size, output_len, topic_len
        topic_embed = torch.bmm(topicattn, topic_embed)
        # batch_size, output_len, embeded_size

        output = torch.cat((topic_embed, output), dim=2)
        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)

        return output

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        # context: batch_size, context_len, 2*enc_hidden_size
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)        
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                
            batch_size, input_len, -1) # batch_topicattentionsize, context_len, dec_hidden_size
        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len 
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1,2)) # batch_size, output_len, context_len
        attn.data.masked_fill(mask, -1e6)
        attn = F.softmax(attn, dim=2) # batch_size, output_len, context_len
        context = torch.bmm(attn, context) # batch_size, output_len, enc_hidden_size * 2
        output = torch.cat((context, output), dim=2) # batch_size, output_len, enc_hidden_size*2 + dec_hidden_size
    
        output = output.view(batch_size*output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dec_layer, topic_num, embed, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = embed
        self.topic_num = topic_num
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.topicattention = TopicAttn(embed_size, enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, dec_layer, batch_first=True)
        self.topic_out = nn.Linear(dec_hidden_size, topic_num)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.emotion_out = External_Memory(dec_hidden_size)
        self.grammar = Filter(dec_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = (torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]).float()
        y_mask = (torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]).float()
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask
    
    def forward(self, ctx, ctx_lengths, tgt, tgt_lengths, hid, topic_embed, topics, tgt_label, emotion, grammar):
        sorted_len, sorted_idx = tgt_lengths.sort(0, descending=True)
        y_sorted = tgt[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]
        
        y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size
        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(tgt_lengths, ctx_lengths)
        output, attn = self.attention(output_seq, ctx, mask) # batch_size, output_len, vocab_size
        output = self.out(output)

        topic_lengths = torch.ones_like(ctx_lengths) * (self.topic_num+1)
        topic_mask = self.create_mask(tgt_lengths, topic_lengths)
        topics_output = self.topicattention(output_seq, topic_embed, topic_mask) # batch_size, output_len, K
        topics_output = self.topic_out(topics_output)
       
        batch_size = output.size(0)
        out_len = output.size(1)
        vocab_size = output.size(2)
        topics = topics.unsqueeze(1).repeat(1, out_len, 1)
        topics_output = torch.zeros(batch_size, out_len, vocab_size, device=output.device).scatter_(2, topics, topics_output)

        output = output + topics_output
        output = self.emotion_out(hid, output, tgt_label, emotion)
        output = self.grammar(output, hid, grammar)
        out = F.log_softmax(output, -1)
        
        return out, hid, attn

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
        self.decoder = Decoder(self.vocab_size, config.embed_size, config.enc_hidden_size, config.dec_hidden_size, config.dec_layer, config.topic_num, embed, config.dropout)
        self.pad_size = config.pad_size
        self.device = config.device
        self.topic_num = config.topic_num
        
    def forward(self, Queries, Responses, emotion=None, grammar=None):
        src = Queries[0]
        tgt = Responses[0]
        batch_size = src.shape[0]
        src_lengths = Queries[1]
        tgt_lengths = Responses[1]
        tgt_label = Responses[2]
        topics = Queries[3]

        tgt = torch.cat((torch.ones(batch_size, 1, device=src.device, dtype=torch.long) * 1, tgt), dim=-1)[:, :self.pad_size]
        encoder_out, hid, topic_embed = self.encoder(src, src_lengths, topics)
        output, hid, attn = self.decoder(encoder_out, src_lengths,
                    tgt, tgt_lengths, hid, 
                    topic_embed, topics, tgt_label, emotion, grammar)
        add = torch.zeros(output.shape[0], self.pad_size-output.shape[1], output.shape[2]).to(self.device)
        out = torch.cat((output, add), dim = 1)
    
        return out, attn

    def response(self, Queries, tgt_labels, emotion=None, grammar=None, max_length=32, beam_width=2):
        src = Queries[0]
        src_lengths = Queries[1]
        topics = Queries[3]
        batch_size = src.shape[0]
        encoder_outs, hids, topic_embeds = self.encoder(src, src_lengths, topics)
        decoded_batch = []
        
        for idx in range(batch_size):
            encoder_out = encoder_outs[idx, :, :].unsqueeze(0)
            hid = hids[:, idx, :].unsqueeze(1)
            topic_embed = topic_embeds[idx, :, :].unsqueeze(0)
            topic = topics[idx,:].unsqueeze(0)
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
                        # if node.wordId in topic:
                        #     reward = 1
                        # elif node.wordId in grammar:
                        #     reward = -1
                        end_nodes.append((node.score(), node))
                        continue

                    output, hid, attn = self.decoder(encoder_out, src_lengths, tgt,
                            torch.ones(1).long().to(tgt.device),
                            hid, topic_embed, topic, tgt_label, emotion, grammar)
                    log_prob, indexes = torch.topk(output.squeeze(0), dim=-1, k=beam_width)
                    
                    for k in range(beam_width):
                        index = indexes[0][k].view(1, -1)
                        log_p = log_prob[0][k]
                        child = BeamSearchNode(hid, node, index, node.logp+log_p, node.length+1)

                        # if child.wordId in topic:
                        #     reward = 1
                        # elif child.wordId in grammar:
                        #     reward = -1
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

    def greedy_response(self, Queries, tgt_label, emotion=None, grammar=None, max_length=30):
        src = Queries[0]
        src_lengths = Queries[1]
        topics = Queries[3]
       
        batch_size = src.shape[0]
        preds = []

        encoder_out, hid, topic_embed = self.encoder(src, src_lengths, topics)
        tgt = torch.Tensor([[1] for i in range(batch_size)]).long().to(self.device)
        for i in range(max_length):
            output, hid, attn = self.decoder(encoder_out, src_lengths, tgt,
                    torch.ones(batch_size).long().to(tgt.device),
                    hid, topic_embed, topics, grammar)

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
