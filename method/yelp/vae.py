from numpy import copy
from utils import Headline
from siamese import print_and_log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import copy

from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from data import get_cuda


def VAE_criterion(x_rec, x, mu, logvar, id_pad=None):
    REC = get_cuda(nn.CrossEntropyLoss(ignore_index=id_pad))(x_rec, x)
    KL = -0.5 * torch.sum(1 - logvar.exp() - mu.pow(2) + logvar)
    return REC + KL


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) +
                              dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        r'''
        src = [src_len(with padding), batch_size]
        src_len = [batch_size]
        '''
        src = src.transpose(0, 1)  # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src)).transpose(
            0, 1)  # embedded = [src_len, batch_size, emb_dim]

        packed = pack_padded_sequence(
            embedded, src_len, batch_first=False, enforce_sorted=False)

        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        # if h_0 is not give, it will be set 0 acquiescently
        enc_output_packed, enc_hidden = self.rnn(packed)
        enc_output, _ = pad_packed_sequence(
            enc_output_packed, batch_first=False)
        # enc_output, enc_hidden = self.rnn(packed)

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...], enc_output are always from the last layer
        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards, encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(
            self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        return enc_output, s


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1]
        embedded = self.dropout(self.embedding(dec_input)).transpose(
            0, 1)  # embedded = [1, batch_size, emb_dim]

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)
        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)
        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))
        return pred, dec_hidden.squeeze(0)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, hid_dim, LATENT_NUM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mu = nn.Linear(hid_dim, LATENT_NUM)
        self.logvar = nn.Linear(hid_dim, LATENT_NUM)
        self.fc = nn.Linear(LATENT_NUM, hid_dim)
        self.hid_dim = hid_dim
        self.LATENT_NUM = LATENT_NUM

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [batch_size, src_len], trg = [batch_size, trg_len]
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = get_cuda(torch.zeros(trg_len, batch_size, trg_vocab_size))
        # enc_output is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        enc_output, hidden = self.encoder(src, src_len)

        hidden, mu, logvar = self.re_hidden(hidden)

        dec_input = trg[0, :]

        for t in range(1, trg_len):
            dec_output, hidden = self.decoder(dec_input, hidden, enc_output)
            outputs[t] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = trg[t] if teacher_force else top1

        # outputs = [trg_len, batch_size, trg_vocab_size]
        return outputs, mu, logvar

    def re_hidden(self, hidden):
        mu = self.mu(hidden.view(-1, self.hid_dim))
        logvar = self.logvar(hidden.view(-1, self.hid_dim))
        z = self.reparameterize(mu, logvar)
        re_hid = self.fc(z).view(-1, self.hid_dim)
        return re_hid, mu, logvar

    def reparameterize(self, mu, logvar):
        epislon = get_cuda(torch.randn(mu.size(0), mu.size(1)))
        z = torch.exp(logvar / 2) * epislon + mu
        return z

    def encoder_only(self, src, src_len):
        with torch.no_grad():
            src = src.transpose(0, 1)
            enc_output, hidden = self.encoder(src, src_len)
        # enc_output = [src_len, 1, enc_hid_dim * 2]
        return enc_output, hidden

    def decoder_only(self, trg, enc_output, hidden, teacher_forcing_ratio=1):
        # trg = [1, trg_len]
        trg = trg.transpose(0, 1)
        dec_input = trg[0, :]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = get_cuda(torch.zeros(trg_len, 1, trg_vocab_size))
        for t in range(1, trg_len):
            dec_output, hidden = self.decoder(dec_input, hidden, enc_output)
            outputs[t] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = trg[t] if teacher_force else top1

        # outputs = [trg_len, 1, trg_vocab_size]
        return outputs


def make_VAE_model(args):
    INPUT_DIM = args.vocab_size
    OUTPUT_DIM = args.vocab_size
    ENC_EMB_DIM = args.encoder_emb_dim
    DEC_EMB_DIM = args.decoder_emb_dim
    ENC_HID_DIM = args.encoder_hidden_dim
    DEC_HID_DIM = args.decoder_hidden_dim
    ENC_DROPOUT = args.encoder_dropout
    DEC_DROPOUT = args.decoder_dropout
    LATENT_NUM = args.AE_LATENT_NUM

    attn = get_cuda(Attention(ENC_HID_DIM, DEC_HID_DIM))
    enc = get_cuda(Encoder(INPUT_DIM, ENC_EMB_DIM,
                           ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT))
    dec = get_cuda(Decoder(OUTPUT_DIM, DEC_EMB_DIM,
                           ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn))

    return get_cuda(VAE(enc, dec, ENC_HID_DIM, LATENT_NUM))


def train_vae(ae_model: VAE, train_data_loader: DataLoader, args):
    ae_model.train()
    criterion = VAE_criterion
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-4)

    print_and_log(Headline('Training VAE Model'))

    for epoch in range(args.max_epoch):
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_data_loader)):
            zero_labels, zero_src, zero_tgt, zero_tgt_y, zero_src_len,\
                one_labels, one_src, one_tgt, one_tgt_y, one_src_len = batch
            r"""
            src = [batch, src_len]
            tgt = [batch, tgt_len]
            """

            src = torch.cat([zero_src, one_src])
            tgt = torch.cat([zero_tgt, one_tgt])
            src_len = torch.cat([zero_src_len, one_src_len])
            
            src = get_cuda(src)
            tgt = get_cuda(tgt)
            src_len = get_cuda(src_len)

            tgt_fu = copy.deepcopy(tgt)
            tgt_fu = tgt_fu.transpose(0, 1)     # tgt_fu = [tgt_len, batch]

            pred, mu, logvar = ae_model(src, src_len, tgt)
            pred_dim = pred.shape[-1]

            loss = criterion(pred[1:].contiguous(
            ).view(-1, pred_dim), tgt_fu[1:].contiguous().view(-1), mu, logvar, id_pad=args.id_pad)

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            epoch_loss += loss.item()

        print_and_log(
            Headline('Loss in epoch.{} is {}'.format(epoch, epoch_loss)))
        torch.save(ae_model.state_dict(), args.current_save_path+'vae_model.pkl')
