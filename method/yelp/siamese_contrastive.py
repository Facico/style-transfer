from argparse import Namespace
import random
from typing import List
# from vae import VAE
import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from model import EncoderDecoder, make_model, Classifier, NoamOpt, LabelSmoothing, fgim_attack
from data import AttackDataLoader, create_data_loader, prepare_data, non_pair_data_loader, get_cuda, \
    pad_batch_seuqences, id2text_sentence, \
    to_var, calc_bleu, load_human_answer, use_cuda
from utils import Headline

from tqdm import trange, tqdm
import copy
import logging
import time
import os


class SiameseNet(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size, latent_size * 2)
        self.norm1 = nn.BatchNorm1d(latent_size * 2)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(latent_size * 2, latent_size * 2)
        self.norm2 = nn.BatchNorm1d(latent_size * 2)
        self.relu2 = nn.ReLU()
        self.d2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(latent_size * 2, latent_size)
        # self.norm3 = nn.BatchNorm1d(latent_size, affine=False)

    def forward(self, x):
        if x == None:
            return None

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.d1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.d2(x)

        x = self.fc3(x)
        return x
        # return self.norm3(x)


class Predictor(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size, 100)
        self.norm1 = nn.BatchNorm1d(100)
        self.relu1 = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(100, latent_size)

    def forward(self, x):
        r"""
        return a detached tensor
        """
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        return x


class Accumulator:
    def __init__(self, name, value: int = 0) -> None:
        self.name = name
        self.value = value
        self.tot_value = value
        self.op_times = 0

    def inc(self, inc_value: int = 1, inc_op_times: int = 1):
        self.tot_value += inc_value
        self.op_times += inc_op_times
        self.value = self.tot_value / self.op_times

    def refresh(self, set_value: int = 0):
        self.value = set_value
        self.tot_value = set_value
        self.op_times = 0

    def __repr__(self) -> str:
        return '{} = {}'.format(self.name, self.value)


def test_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args, cur_epoch=None):
    r"""
    Testing Siamese Net
    """
    # zero_data_loader, one_data_loader = create_test_data_loader(
    #     args, args.batch_size)
    zero_data_loader, one_data_loader = create_test_data_loader(
        args, 1)

    ae_model.eval()
    siamese_net.eval()
    predictor_net.eval()

    with torch.no_grad():

        criterion = get_cuda(nn.CosineSimilarity())

        Loss_Accumulator = Accumulator('Loss')
        Acc_diff_Accumulator = Accumulator('Diff Accuracy')
        Acc_sim_Accumulator = Accumulator('Sim Accuracy')

        print(Headline('Test in epoch {}'.format(cur_epoch)))
        if None != cur_epoch:
            logging.info(Headline('Testing in epoch{}'.format(cur_epoch)))
        else:
            logging.info(Headline('Testing'))

        for it in range(zero_data_loader.num_batch):
            zero_batch_sentences, zero_tensor_labels, \
                zero_tensor_src, zero_tensor_src_mask, zero_tensor_tgt, zero_tensor_tgt_y, \
                zero_tensor_tgt_mask, _ = zero_data_loader.next_batch()

            one_batch_sentences, one_tensor_labels, \
                one_tensor_src, one_tensor_src_mask, one_tensor_tgt, one_tensor_tgt_y, \
                one_tensor_tgt_mask, _ = one_data_loader.next_batch()

            # Forward Pass
            zero_latent, _ = ae_model.forward(
                zero_tensor_src, zero_tensor_tgt, zero_tensor_src_mask, zero_tensor_tgt_mask)
            one_latent, _ = ae_model.forward(
                one_tensor_src, one_tensor_tgt, one_tensor_src_mask, one_tensor_tgt_mask)

            zero_latent_shuffle = shuffle_tensor(zero_latent)
            one_latent_shuffle = shuffle_tensor(one_latent)

            new_latent = torch.cat([zero_latent, one_latent])
            new_reverse_latent = torch.cat(
                [zero_latent_shuffle, one_latent_shuffle])

            loss = None
            acc = None

            r"""
            Evaluate Same Style Acc
            """
            if args.simplest_siamese:
                z1, z2 = siamese_net(new_latent), siamese_net(
                    new_reverse_latent)
                loss = - criterion(z2, z1).mean()
                acc = (criterion(z1, z2) > 0).sum()
            else:
                z1, z2 = siamese_net(new_latent), siamese_net(
                    new_reverse_latent)
                p1, p2 = predictor_net(z1), predictor_net(z2)
                loss = - (criterion(p1, z2.detach()) +
                          criterion(p2, z1.detach())).mean() / 2
                acc = ((criterion(z1, p2) > 0).sum() +
                       (criterion(z2, p1) > 0).sum()).float() / 2

            Loss_Accumulator.inc(loss.item())
            Acc_sim_Accumulator.inc(acc.item(), new_latent.size(0))

            r"""
            Evaluate Different Style Acc
            """
            new_latent = torch.cat([zero_latent, zero_latent_shuffle])
            new_reverse_latent = torch.cat(
                [one_latent, one_latent_shuffle])

            new_latent = zero_latent
            new_reverse_latent = one_latent

            z1, z2 = siamese_net(new_latent), siamese_net(new_reverse_latent)
            if args.simplest_siamese:
                acc = (criterion(z1, z2) < 0).sum()
            else:
                p1, p2 = predictor_net(z1), predictor_net(z2)
                c1 = (criterion(z1, p2))
                c2 = (criterion(z2, p1))
                acc = ((c1 < 0).sum() + (c2 < 0).sum()).float() / 2

            Acc_diff_Accumulator.inc(acc.item(), new_latent.size(0))

            # We cannot run so many test cases on CPU
            if it > 100 and not use_cuda:
                break

        print_and_log(str(Acc_sim_Accumulator))
        print_and_log(str(Acc_diff_Accumulator))
        print_and_log(str(Loss_Accumulator))


def create_train_data_loader_contrastive(args, batch_size=None, c='train', pos_num=None, neg_num=None):
    if batch_size is None:
        batch_size = args.batch_size

    if_shuffle = not (c == 'attack')

    ori_train_data_loader = AttackDataLoader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    like_train_data_loader = AttackDataLoader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    unlike_train_data_loader = AttackDataLoader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )

    ori_file_list = [args.data_path + 'neg10_pos10_train0.json',
                     args.data_path + 'neg10_pos10_train1.json']
    like_file_list = [args.data_path + 'neg10_pos10_train0.json',
                      args.data_path + 'neg10_pos10_train1.json']
    unilke_file_list = [args.data_path + 'neg10_pos10_train0.json',
                        args.data_path + 'neg10_pos10_train1.json']

    if c == 'attack':
        ori_file_list = [args.data_path + 'test_ref0.json',
                         args.data_path + 'test_ref1.json']
        like_file_list = [args.data_path + 'test_ref0.json',
                          args.data_path + 'test_ref1.json']
        unilke_file_list = [args.data_path + 'test_ref0.json',
                            args.data_path + 'test_ref1.json']

        ori_file_list = [args.data_path + 'neg10_pos10_test0.json',
                         args.data_path + 'neg10_pos10_test1.json']
        like_file_list = [args.data_path + 'neg10_pos10_test0.json',
                          args.data_path + 'neg10_pos10_test1.json']
        unilke_file_list = [args.data_path + 'neg10_pos10_test0.json',
                            args.data_path + 'neg10_pos10_test1.json']

        if args.reverse_order:
            ori_file_list = ori_file_list[::-1]
            like_file_list = like_file_list[::-1]
            unilke_file_list = unilke_file_list[::-1]

    elif c == 'impossible':
        ori_file_list = [args.data_path + 'impossible_ref0.json',
                         args.data_path + 'impossible_ref1.json']
        like_file_list = [args.data_path + 'impossible_ref0.json',
                          args.data_path + 'impossible_ref1.json']
        unilke_file_list = [args.data_path + 'impossible_ref0.json',
                            args.data_path + 'impossible_ref1.json']

    ori_label_list = [[0], [1]]
    like_label_list = [[0], [1]]
    unlike_label_list = [[1], [0]]

    print_and_log(f'Loading {c} Dataset, Shuffle={if_shuffle}')
    print_and_log(f'ori_file_list = {ori_file_list}')

    rand_seed = 234
    print(f'Random Seed is set to {rand_seed}')

    random.seed(rand_seed)
    ori_train_data_loader.create_batches(
        ori_file_list, origin_label='only', if_shuffle=if_shuffle, pos_num=pos_num, neg_num=neg_num
    )
    random.seed(rand_seed)
    like_train_data_loader.create_batches(
        like_file_list, origin_label='like', if_shuffle=if_shuffle, pos_num=pos_num, neg_num=neg_num
    )
    random.seed(rand_seed)
    unlike_train_data_loader.create_batches(
        unilke_file_list, origin_label='not', if_shuffle=if_shuffle, pos_num=pos_num, neg_num=neg_num
    )

    return ori_train_data_loader, like_train_data_loader, unlike_train_data_loader


def create_train_data_loader(args):
    zero_train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    one_train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )

    if args.tfidf_train:
        zero_file_list = [args.data_path + 'similar_train_ref0',
                          args.data_path + 'similar_train0']
        one_file_list = [args.data_path + 'similar_train1',
                         args.data_path + 'similar_train_ref1']
        zero_label_list = [[0], [0]]
        one_label_list = [[1], [1]]

    else:
        zero_file_list = [args.data_path + 'sentiment.train.0',
                          args.data_path + 'sentiment.dev.0']
        one_file_list = [args.data_path + 'sentiment.train.1',
                         args.data_path + 'sentiment.dev.1']
        zero_label_list = [[0], [0]]
        one_label_list = [[1], [1]]

    if_shuffle = not args.tfidf_train

    zero_train_data_loader.create_batches(
        zero_file_list, zero_label_list, if_shuffle=if_shuffle
    )
    one_train_data_loader.create_batches(
        one_file_list, one_label_list, if_shuffle=if_shuffle
    )

    return zero_train_data_loader, one_train_data_loader


def batch_forward(reval_num: int, net: nn.Module, *args, **kwargs):
    reval = None
    if args[0] == None:
        if reval_num == 1:
            return reval

        reval = [None] * reval_num
        return reval
    batch_size = args[0].size(0)
    feed_tensors = list(zip(*args))

    for i in range(batch_size):
        tmp_reval = net(*feed_tensors[i], **kwargs)
        if isinstance(tmp_reval, tuple):
            tuple_len = len(tmp_reval)
            if reval is None:
                reval = [None] * tuple_len
            for j in range(tuple_len):
                if reval[j] is None:
                    reval[j] = tmp_reval[j].unsqueeze(0)
                else:
                    reval[j] = torch.cat(
                        [reval[j], tmp_reval[j].unsqueeze(0)], dim=0)
        else:
            if reval is None:
                reval = tmp_reval.unsqueeze(0)
            else:
                reval = torch.cat([reval, tmp_reval.unsqueeze(0)], dim=0)

    return reval


def get_contrastive_loss(siamese_net: SiameseNet,
                         predictor_net: Predictor,
                         criterion: nn.NLLLoss,
                         ori_latent: Tensor,
                         like_latent: Tensor,
                         unlike_latent: Tensor,
                         args, not_nll=False):
    r"""
    Get the loss of contrastive：log_softmax + nll loss
    """
    neg_num = unlike_latent.shape[0] if unlike_latent != None else 0
    pos_num = like_latent.shape[0] if like_latent != None else 0

    ori_z = siamese_net(ori_latent)
    like_z = batch_forward(1, siamese_net, like_latent)
    unlike_z = batch_forward(1, siamese_net, unlike_latent)

    if args.simplest_siamese:
        # pos_score = torch.bmm(like_z.permute(1, 0, 2),
        #                       ori_z.unsqueeze(2)).squeeze(2)
        # # [batch size, neg num, hidden]*[batch size, hidden, 1] = [batch size, neg num, 1]
        # neg_score = torch.bmm(unlike_z.permute(1, 0, 2),
        #                       ori_z.unsqueeze(2)).squeeze(2)
        if like_z != None:
            pos_score = F.cosine_similarity(
                ori_z, like_z, dim=-1).transpose(0, 1) / args.tau
        else:
            pos_score = None

        if unlike_z != None:
            neg_score = F.cosine_similarity(
                ori_z, unlike_z, dim=-1).transpose(0, 1) / args.tau
        else:
            neg_score = None
    else:
        r"""
        只有origin需要过predictor_net，原来都过predictor_net是因为如果只有两个例子的话就是对称的
        """
        ori_p = predictor_net(ori_z)
        like_z = like_z.detach() if like_z != None else None
        unlike_z = unlike_z.detach() if unlike_z != None else None
        # like_z, unlike_z = like_z.detach(), unlike_z.detach()
        if like_z != None:
            pos_score = F.cosine_similarity(
                ori_p, like_z, dim=-1).transpose(0, 1) / args.tau
        else:
            pos_score = None

        if unlike_z != None:
            neg_score = F.cosine_similarity(
                ori_p, unlike_z, dim=-1).transpose(0, 1) / args.tau
        else:
            neg_score = None

    losses = []
    choice = []

    if not_nll == True:
        pos_loss = - sum(pos_score) if pos_score != None else 0
        neg_loss = sum(neg_score) if neg_score != None else 0

        if pos_loss != None and neg_loss != None:
            loss = pos_loss + neg_loss
        else:
            loss = pos_loss if pos_loss else neg_loss

        if pos_loss != None and neg_loss != None:
            return sum(loss) / (neg_num+pos_num), ((pos_loss < 0).sum() + (neg_loss < 0).sum()).item()
        else:
            return sum(loss) / (neg_num+pos_num), None

    for i in range(pos_num):
        score = torch.cat([pos_score[:, i].unsqueeze(-1), neg_score],
                          1)  # [batch size, 1 + neg num]
        targets = get_cuda(torch.tensor([0] * score.size(0)))
        score_softmax = F.log_softmax(score, dim=1)
        choice.append(score_softmax.argmax(
            dim=-1).clone().cpu().numpy().tolist())
        cur_loss = criterion(score_softmax, targets)
        losses.append(cur_loss)

    loss = sum(losses) / (pos_num)
    loss_just_one = min(losses)

    if (args.just_use_one):
        return loss_just_one, choice
    return loss, choice


def get_siamese_loss(siamese_net: SiameseNet,
                     predictor_net: Predictor,
                     criterion: nn.CosineSimilarity,
                     zero_latent: Tensor,
                     one_latent: Tensor,
                     args):
    r"""
    When Entering this function, make sure `siamese_net` and `predictor_net` are in train mod, not eval
    """

    zero_latent_shuffle, one_latent_shuffle = shuffle_tensor(
        zero_latent, one_latent)

    latent1, latent2, targets, loss = None, None, None, None

    if args.negative_samples:
        latent1 = torch.cat(
            [zero_latent, one_latent, zero_latent_shuffle, one_latent_shuffle])
        latent2 = torch.cat(
            [zero_latent_shuffle, one_latent_shuffle, one_latent, zero_latent])
        targets = get_cuda(torch.tensor(
            [1] * 2 * zero_latent.size(0) + [-1] * 2 * zero_latent.size(0)).float())
    else:
        latent1 = torch.cat([zero_latent, one_latent])
        latent2 = torch.cat([zero_latent_shuffle, one_latent_shuffle])
        targets = get_cuda(torch.tensor([1] * latent1.size(0)).float())

    z1, z2 = siamese_net(latent1), siamese_net(latent2)

    if args.simplest_siamese:
        loss = - (criterion(z2, z1) * targets).mean()
    else:
        p1, p2 = predictor_net(z1), predictor_net(z2)
        z1, z2 = z1.detach(), z2.detach()
        loss1 = (targets * criterion(p1, z2)).mean()
        loss2 = (targets * criterion(p2, z1)).mean()
        loss = - (loss1 + loss2) / 2

    return loss


def train_ae_model(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args,
                   epoch_num: int = 1):
    print_and_log(Headline('Training the whole model'))

    ae_model.train()
    siamese_net.train()
    predictor_net.train()

    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    params = list(siamese_net.parameters()) + \
        list(predictor_net.parameters())
    optimizer = optim.Adam(params, lr=args.train_siamese_lr)

    criterion = get_cuda(nn.CosineSimilarity())
    ae_criterion = get_cuda(LabelSmoothing(
        size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))

    zero_train_data_loader, one_train_data_loader = create_train_data_loader(
        args)

    for epoch in range(epoch_num):
        print_and_log(
            Headline('Start Training Every thing at Epoch.{}'.format(epoch)))
        epoch_loss = 0
        for it in trange(zero_train_data_loader.num_batch):
            zero_batch_sentences, zero_tensor_labels, \
                zero_tensor_src, zero_tensor_src_mask, zero_tensor_tgt, zero_tensor_tgt_y, \
                zero_tensor_tgt_mask, zero_tensor_ntokens = zero_train_data_loader.next_batch()

            one_batch_sentences, one_tensor_labels, \
                one_tensor_src, one_tensor_src_mask, one_tensor_tgt, one_tensor_tgt_y, \
                one_tensor_tgt_mask, one_tensor_ntokens = one_train_data_loader.next_batch()

            zero_latent, zero_out = ae_model.forward(
                zero_tensor_src, zero_tensor_tgt, zero_tensor_src_mask, zero_tensor_tgt_mask)

            one_latent, one_out = ae_model.forward(
                one_tensor_src, one_tensor_tgt, one_tensor_src_mask, one_tensor_tgt_mask)

            # Reconstruction Loss
            zero_loss_rec = ae_criterion(zero_out.contiguous().view(-1, zero_out.size(-1)),
                                         zero_tensor_tgt_y.contiguous().view(-1)) / zero_tensor_ntokens.data
            one_loss_rec = ae_criterion(one_out.contiguous().view(-1, one_out.size(-1)),
                                        one_tensor_tgt_y.contiguous().view(-1)) / one_tensor_ntokens.data
            loss_rec = zero_loss_rec + one_loss_rec

            # Siamese loss, same as in function `train_siamese`
            siamese_loss = get_siamese_loss(siamese_net,
                                            predictor_net,
                                            criterion,
                                            zero_latent,
                                            one_latent,
                                            args)

            ################################################################################
            # Finish Loss Calculation
            ################################################################################
            anneal_ratio = min(1.0, epoch / args.anneal_epoch)

            loss = loss_rec + anneal_ratio * siamese_loss

            epoch_loss += loss.item()

            ae_optimizer.optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            if it % 200 == 0:
                logging.info(
                    'epoch {}, batch {}, loss={}'.format(epoch, it, loss))
                # print_latent_sent(zero_latent[0], ae_model, args)

        print_and_log('Epoch loss = {}'.format(epoch_loss))
        torch.save(ae_model.state_dict(), args.current_save_path +
                   'new_ae_model_params.pkl')
        # save_siamese(siamese_net, predictor_net, args)
        # test_siamese(siamese_net, predictor_net, ae_model, args, epoch)

        ae_model.train()
        siamese_net.train()
        predictor_net.train()


def shuffle_tensor(*args):
    r"""
    Shuffle a tensor
    """
    r = torch.randperm(args[0].size(0))
    return [t[r] for t in args] if len(args) != 1 else args[0][r]


def train_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args,
                  epoch_num: int = 1):
    r"""
    Training Siamese Net
    """
    zero_train_data_loader, one_train_data_loader = create_train_data_loader(
        args)

    ae_model.eval()
    siamese_net.train()
    predictor_net.train()

    optimizer = None

    if args.simplest_siamese:
        optimizer = optim.Adam(siamese_net.parameters(),
                               lr=args.train_siamese_lr)
    else:
        params = list(siamese_net.parameters()) + \
            list(predictor_net.parameters())
        optimizer = optim.Adam(params, lr=args.train_siamese_lr)

    criterion = get_cuda(nn.CosineSimilarity())

    print_and_log(Headline('Training Siamese by lr={}'.format(
        args.train_siamese_lr), note_token='+'))

    for epoch in range(epoch_num):
        logging.info(Headline('Start Running Epoch {}'.format(epoch)))
        print(Headline('Start Running Epoch {}'.format(epoch)))
        for it in trange(zero_train_data_loader.num_batch):
            zero_batch_sentences, zero_tensor_labels, \
                zero_tensor_src, zero_tensor_src_mask, zero_tensor_tgt, zero_tensor_tgt_y, \
                zero_tensor_tgt_mask, zero_tensor_ntokens = zero_train_data_loader.next_batch()

            one_batch_sentences, one_tensor_labels, \
                one_tensor_src, one_tensor_src_mask, one_tensor_tgt, one_tensor_tgt_y, \
                one_tensor_tgt_mask, one_tensor_ntokens = one_train_data_loader.next_batch()

            r"""
            tensor_labels = [128, 1]
            tensor_src = [128, 15]
            tensor_src_mask = [128, 1, 15]
            tensor_tgt = [128, 16], with one more <bos>
            tensor_tgt_y = [128, 16], with one more <eos>
            tensor_tgt_mask = [128, 16, 16]
            tensor_ntokens = [], a number
            """

            # Forward Pass
            zero_latent, _ = ae_model.forward(
                zero_tensor_src, zero_tensor_tgt, zero_tensor_src_mask, zero_tensor_tgt_mask)

            one_latent, _ = ae_model.forward(
                one_tensor_src, one_tensor_tgt, one_tensor_src_mask, one_tensor_tgt_mask)

            r"""
            latent = [128, 256]
            out = [128, 16, 9339]
            """
            loss = get_siamese_loss(siamese_net,
                                    predictor_net,
                                    criterion,
                                    zero_latent,
                                    one_latent,
                                    args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 200 == 0:
                logging.info(
                    'epoch {}, batch {}, loss={}'.format(epoch, it, loss))

        test_siamese(siamese_net, predictor_net, ae_model, args, epoch)
        evaluate_attack_dataset_acc(siamese_net, predictor_net, ae_model, args)
        save_siamese(siamese_net, predictor_net, args)

        siamese_net.train()
        predictor_net.train()


def attack_contrastive(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args):
    pos_num = args.pos_num
    neg_num = args.neg_num

    ori_attack_data_loader, like_attack_data_loader, unlike_attack_data_loader = create_train_data_loader_contrastive(
        args, 1, 'attack', pos_num=pos_num, neg_num=neg_num)

    print_and_log(Headline('Start Attack via Contrastive Net'))
    print_and_log(Headline('Optimizer {}, lr {}, Iter {}'.format(
        args.victim_optimizer, args.victim_lr, args.attack_max_iter)))
    if args.simplest_siamese:
        print_and_log(
            Headline('Attack with Simplest Siamest Net', note_token='+'))
    else:
        print_and_log(
            Headline('Attack with enhance Siamese Net(with Predictor)', note_token='+'))

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()

    c = copy.deepcopy

    for it in range(ori_attack_data_loader.num_batch):
        ori_src, ori_tgt, ori_src_mask, ori_tgt_mask = ori_attack_data_loader.next_batch()

        like_src, like_tgt, like_src_mask, like_tgt_mask = like_attack_data_loader.next_batch()

        unlike_src, unlike_tgt, unlike_src_mask, unlike_tgt_mask = unlike_attack_data_loader.next_batch()

        print_and_log(Headline('Sentence Pair.{}'.format(it)))
        print_and_log('Src: ' + id2text_sentence(
            ori_src[0], args.id_to_word))

        ori_latent, _ = ae_model.forward(
            ori_src, ori_tgt, ori_src_mask, ori_tgt_mask)
        like_latent, _ = batch_forward(2,
                                       ae_model, like_src, like_tgt, like_src_mask, like_tgt_mask)
        unlike_latent, _ = batch_forward(2,
                                         ae_model, unlike_src, unlike_tgt, unlike_src_mask, unlike_tgt_mask)

        ori_generated_text = ae_model.greedy_decode(ori_latent,
                                                    max_len=args.max_sequence_length,
                                                    start_id=args.id_bos)
        like_generated_text = batch_forward(1,
                                            ae_model.greedy_decode, like_latent, max_len=args.max_sequence_length, start_id=args.id_bos)
        unlike_generated_text = batch_forward(1,
                                              ae_model.greedy_decode, unlike_latent, max_len=args.max_sequence_length, start_id=args.id_bos)

        print_and_log('Rec: ' + id2text_sentence(
            ori_generated_text[0], args.id_to_word))

        if like_latent != None:
            for i in range(like_latent.size(0)):
                print_and_log('Sim{}: '.format(i) + id2text_sentence(
                    like_generated_text[i].squeeze(), args.id_to_word))

        if unlike_latent != None:
            for i in range(unlike_latent.size(0)):
                print_and_log('Ref{}: '.format(i) + id2text_sentence(
                    unlike_generated_text[i].squeeze(), args.id_to_word))

        # contrastive_attack_one(siamese_net, predictor_net, ae_model, c(ori_latent.detach()),
        #    c(unlike_latent.detach()), c(like_latent.detach()), args)
        like_latent = c(like_latent.detach()) if like_latent != None else None
        unlike_latent = c(unlike_latent.detach()
                          ) if unlike_latent != None else None

        args.victim_lr = args.pos_victim_lr if it < 500 else args.neg_victim_lr

        contrastive_attack_one(siamese_net, predictor_net, ae_model, c(ori_latent.detach()),
                               like_latent, unlike_latent, args)


def text_to_tensor(text: str, args):
    text_list = text.split()
    id_list = [args.word_to_id[w] for w in text_list]
    src = get_cuda(torch.tensor(id_list, dtype=torch.long).unsqueeze(0))
    src_mask = (src != 0).unsqueeze(-2)
    return src, src_mask


def contrastive_attack_one(siamese_net: SiameseNet,
                           predictor_net: Predictor,
                           ae_model: EncoderDecoder,
                           ori_latent: torch.Tensor,
                           like_latent: torch.Tensor,
                           unlike_latent: torch.Tensor,
                           args):
    r"""
    ori_latent = [1, sent_len]
    like_latent = [like_num, 1, sent_len]
    unlike_latent = [unlike_num, 1, sent_len]
    """
    ori_latent.requires_grad = True
    if like_latent != None:
        like_latent.requires_grad = False
        like_num = like_latent.size(0)
    else:
        like_num = 0

    if unlike_latent != None:
        unlike_latent.requires_grad = False
        unlike_num = unlike_latent.size(0)
    else:
        unlike_num = 0

    criterion = get_cuda(nn.NLLLoss())
    ori_optimizer = getattr(optim, args.victim_optimizer)(
        [ori_latent], lr=args.victim_lr)

    ori_generated_text = ae_model.greedy_decode(ori_latent,
                                                max_len=args.max_sequence_length,
                                                start_id=args.id_bos)
    ori_generator_text = id2text_sentence(
        ori_generated_text[0], args.id_to_word)

    print_and_log(Headline('', note_token='-'))

    change_iter = 0
    if args.dance:
        args.attack_max_iter = 2

    for it in range(args.attack_max_iter):
        loss, dance = get_contrastive_loss(
            siamese_net,
            predictor_net,
            criterion,
            ori_latent,
            like_latent,
            unlike_latent,
            args,
            not_nll=args.not_nll)

        loss = -(loss)

        ori_optimizer.zero_grad()
        loss.backward()
        ori_optimizer.step()

        generated_text = ae_model.greedy_decode(ori_latent,
                                                max_len=args.max_sequence_length,
                                                start_id=args.id_bos)

        generator_text = id2text_sentence(
            generated_text[0], args.id_to_word)

        if args.retext:
            re_generator_text, re_latent = retext(
                generator_text, ae_model, args)
        else:
            re_generator_text, re_latent = generator_text, ori_latent

        if ori_generator_text != re_generator_text and re_generator_text == generator_text:
            ori_latent.data = re_latent.data
            ori_generator_text = re_generator_text
            if not args.dance:
                print_and_log('Change.{}, {}'.format(
                    change_iter, ori_generator_text))
                change_iter += 1

        if args.dance:
            print_and_log('Siamese Loss score:{}'.format(loss.item()))
            print_and_log('Siamese attack tensors:{}'.format(dance))
        print_and_log('iteration.{}, {}'.format(it, ori_generator_text))

        if change_iter >= args.attack_max_change:
            break


def retext(ori_text: str,
           ae_model: EncoderDecoder,
           args):

    with torch.no_grad():
        new_src, new_src_mask = text_to_tensor(ori_text, args)
        ori_latent = ae_model.get_latent(new_src, new_src_mask)
        ori_generated_text = ae_model.greedy_decode(ori_latent,
                                                    max_len=args.max_sequence_length,
                                                    start_id=args.id_bos)
        ori_generator_text = id2text_sentence(
            ori_generated_text[0], args.id_to_word)

        return ori_generator_text, ori_latent


def train_contrastive(siamese_net: SiameseNet,
                      predictor_net: Predictor,
                      ae_model: EncoderDecoder,
                      args, epoch_num: int = 1):
    r"""
    Training Contrastive Learning
    """
    pos_num = args.pos_num
    neg_num = args.neg_num

    ori_train_data_loader, like_train_data_loader, unlike_train_data_loader = create_train_data_loader_contrastive(
        args, pos_num=pos_num, neg_num=neg_num)

    ae_model.eval()
    siamese_net.train()
    predictor_net.train()

    optimizer = None

    if args.simplest_siamese:
        optimizer = optim.Adam(siamese_net.parameters(),
                               lr=args.train_siamese_lr)
    else:
        params = list(siamese_net.parameters()) + \
            list(predictor_net.parameters())
        optimizer = optim.Adam(params, lr=args.train_siamese_lr)

    criterion = get_cuda(nn.NLLLoss())

    print_and_log(Headline('Training Siamese by lr={}'.format(
        args.train_siamese_lr), note_token='+'))

    for epoch in range(epoch_num):
        logging.info(Headline('Start Running Epoch {}'.format(epoch)))
        print(Headline('Start Running Epoch {}'.format(epoch)))
        epoch_loss = 0
        cur_acc, cur_num = 0, 0
        pbar = tqdm(range(ori_train_data_loader.num_batch))
        for it in pbar:
            ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask = ori_train_data_loader.next_batch()

            like_tensor_src, like_tensor_tgt, like_tensor_src_mask, like_tensor_tgt_mask = like_train_data_loader.next_batch()

            unlike_tensor_src, unlike_tensor_tgt, unlike_tensor_src_mask, unlike_tensor_tgt_mask = unlike_train_data_loader.next_batch()

            # Forward Pass
            ori_latent, _ = ae_model.forward(
                ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask)

            # like = [pos num, batch size, hidden]
            like_latent, _ = batch_forward(2,
                                           ae_model, like_tensor_src, like_tensor_tgt, like_tensor_src_mask, like_tensor_tgt_mask)

            # unlike = [neg num, batch size, hidden]
            unlike_latent, _ = batch_forward(2,
                                             ae_model, unlike_tensor_src, unlike_tensor_tgt, unlike_tensor_src_mask, unlike_tensor_tgt_mask)

            ori_latent, like_latent, unlike_latent = ori_latent.detach(
            ), like_latent.detach(), unlike_latent.detach()

            loss, choice = get_contrastive_loss(siamese_net,
                                                predictor_net,
                                                criterion,
                                                ori_latent,
                                                like_latent,
                                                unlike_latent,
                                                args)

            choice = np.array(choice).flatten()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_num += len(choice)
            cur_acc += sum(choice == 0)

            if it % 200 == 0:
                pbar.set_description('batch {}, loss={:.3f}, acc={:.3f}'.format(
                    it, loss, cur_acc / cur_num))
                # print_and_log(
                # 'epoch {}, batch {}, loss={}, acc={}'.format(epoch, it, loss, cur_acc/cur_num))

        # test_siamese(siamese_net, predictor_net, ae_model, args, epoch)
        # evaluate_attack_dataset_acc(siamese_net, predictor_net, ae_model, args)
        print_and_log(f'Epoch Loss={epoch_loss}')
        save_siamese(siamese_net, predictor_net, args)

        siamese_net.train()
        predictor_net.train()


def create_ae_data_loader(args, batch_size=None, pos_num=None, neg_num=None):
    if batch_size is None:
        batch_size = args.batch_size

    if_shuffle = True

    ori_train_data_loader = AttackDataLoader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    like_train_data_loader = AttackDataLoader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )

    ori_file_list = [args.data_path + 'ae_train_constraint0.json',
                     args.data_path + 'ae_train_constraint1.json']
    like_file_list = [args.data_path + 'ae_train_constraint0.json',
                      args.data_path + 'ae_train_constraint1.json']

    # ori_file_list = [args.data_path + 'sbert_train_ref0.json',
                     # args.data_path + 'sbert_train_ref1.json']
    # like_file_list = [args.data_path + 'sbert_train_ref0.json',
                      # args.data_path + 'sbert_train_ref1.json']

    # ori_file_list = [args.data_path + 'test_ref0.json',
    #                  args.data_path + 'test_ref1.json']
    # like_file_list = [args.data_path + 'test_ref0.json',
    #                   args.data_path + 'test_ref1.json']

    print_and_log(f'Loading Dataset, Shuffle={if_shuffle}')
    print_and_log(f'ori_file_list = {ori_file_list}')

    rand_seed = 234
    print(f'Random Seed is set to {rand_seed}')

    random.seed(rand_seed)
    ori_train_data_loader.create_batches(
        ori_file_list, origin_label='only', if_shuffle=if_shuffle, pos_num=pos_num, neg_num=neg_num
    )
    random.seed(rand_seed)
    like_train_data_loader.create_batches(
        like_file_list, origin_label='like', if_shuffle=if_shuffle, pos_num=pos_num, neg_num=neg_num
    )

    return ori_train_data_loader, like_train_data_loader


def train_ae_model_constraint(ae_model: EncoderDecoder,
                              args, epoch_num: int = 1):
    r"""
    Training Contrastive Learning
    """
    ori_train_data_loader, like_train_data_loader = create_ae_data_loader(args)

    ae_model.train()

    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    ae_criterion = get_cuda(LabelSmoothing(
        size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    optimizer = None

    criterion = get_cuda(nn.NLLLoss())

    print_and_log(
        Headline('Training ae_model with Constraint', note_token='+'))

    for epoch in range(epoch_num):
        logging.info(Headline('Start Running Epoch {}'.format(epoch)))
        print(Headline('Start Running Epoch {}'.format(epoch)))

        epoch_rec_loss = Accumulator('rec loss')
        epoch_constraint_loss = Accumulator('constraint loss')

        pbar = tqdm(range(ori_train_data_loader.num_batch))
        for it in pbar:
            losses = []
            ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask, ori_tensor_tgt_y, ori_tensor_ntokens = ori_train_data_loader.next_batch(
                type='ae')

            # like_tensor_src = [like_num, batch_size, seq_len]
            like_tensor_src, like_tensor_tgt, like_tensor_src_mask, like_tensor_tgt_mask = like_train_data_loader.next_batch()

            # ae_model Forward pass
            ori_latent, out = ae_model.forward(
                ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask)

            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    ori_tensor_tgt_y.contiguous().view(-1)) / ori_tensor_ntokens.data

            epoch_rec_loss.inc(loss_rec.item())
            # epoch_rec_loss += loss_rec.item()

            ae_optimizer.optimizer.zero_grad()

            if True:
                like_latent, _ = batch_forward(2,
                                               ae_model, like_tensor_src, like_tensor_tgt, like_tensor_src_mask, like_tensor_tgt_mask)

                ori_latent_cp, _ = ae_model.forward(
                    ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask)

                like_latent = torch.cat(
                    [like_latent, ori_latent_cp.unsqueeze(0)], dim=0)
                batch_size = ori_latent.shape[0]
                like_num = like_latent.shape[0]

                for i in range(like_num):
                    sim = F.cosine_similarity(ori_latent.unsqueeze(
                        1), like_latent[i], dim=-1) / args.tau

                    targets = list(range(batch_size))
                    targets = get_cuda(torch.tensor(targets))
                    sim_softmax = F.log_softmax(sim, dim=-1)
                    constraint_loss = criterion(sim_softmax, targets)

                    losses.append(constraint_loss)

                constraint_loss = sum(losses) / len(losses)
                (args.bata * constraint_loss + loss_rec).backward()

                epoch_constraint_loss.inc(constraint_loss.item())
            else:
                (loss_rec).backward()

            ae_optimizer.step()

            if it % 200 == 0:
                pbar.set_description(
                    'batch {}, recon={:.3f}, constraint={:.3f}'.format(it, epoch_rec_loss.value, epoch_constraint_loss.value))

                logging.info(
                    'epoch {}, batch {}, rec loss={}, constraint loss={}'.format(epoch, it, epoch_rec_loss.value, epoch_constraint_loss.value))

        torch.save(ae_model.state_dict(), args.current_save_path +
                   'constraint_ae_model{:d}_params.pkl'.format(epoch))


def train_together(siamese_net: SiameseNet,
                   predictor_net: Predictor,
                   ae_model: EncoderDecoder,
                   args, epoch_num: int = 1):
    r"""
    Training Contrastive Learning
    """
    ori_train_data_loader, like_train_data_loader, unlike_train_data_loader = create_train_data_loader_contrastive(
        args, c='impossible')

    ae_model.train()
    siamese_net.train()
    predictor_net.train()

    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    ae_criterion = get_cuda(LabelSmoothing(
        size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    optimizer = None

    if args.simplest_siamese:
        optimizer = optim.Adam(siamese_net.parameters(),
                               lr=args.train_siamese_lr)
    else:
        params = list(siamese_net.parameters()) + \
            list(predictor_net.parameters())
        optimizer = optim.Adam(params, lr=args.train_siamese_lr)

    criterion = get_cuda(nn.NLLLoss())

    print_and_log(Headline('Training together by lr={}'.format(
        args.train_siamese_lr), note_token='+'))

    for epoch in range(epoch_num):
        logging.info(Headline('Start Running Epoch {}'.format(epoch)))
        print(Headline('Start Running Epoch {}'.format(epoch)))
        epoch_loss = 0
        cur_acc, cur_num = 0, 0
        epoch_rec_loss = 0
        pbar = tqdm(range(ori_train_data_loader.num_batch))
        for it in pbar:
            ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask, ori_tensor_tgt_y, ori_tensor_ntokens = ori_train_data_loader.next_batch(
                type='ae')

            like_tensor_src, like_tensor_tgt, like_tensor_src_mask, like_tensor_tgt_mask = like_train_data_loader.next_batch()

            unlike_tensor_src, unlike_tensor_tgt, unlike_tensor_src_mask, unlike_tensor_tgt_mask = unlike_train_data_loader.next_batch()

            # ae_model Forward pass
            latent, out = ae_model.forward(
                ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask)

            # Loss calculation
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    ori_tensor_tgt_y.contiguous().view(-1)) / ori_tensor_ntokens.data

            epoch_rec_loss += loss_rec.item()

            ae_optimizer.optimizer.zero_grad()

            loss_rec.backward()
            ae_optimizer.step()

            if ((it + 1) % 3 != 0):
                continue
            # contrastive Forward Pass
            ori_latent, _ = ae_model.forward(
                ori_tensor_src, ori_tensor_tgt, ori_tensor_src_mask, ori_tensor_tgt_mask)

            # like = [pos num, batch size, hidden]
            like_latent, _ = batch_forward(2,
                                           ae_model, like_tensor_src, like_tensor_tgt, like_tensor_src_mask, like_tensor_tgt_mask)

            # unlike = [neg num, batch size, hidden]
            unlike_latent, _ = batch_forward(2,
                                             ae_model, unlike_tensor_src, unlike_tensor_tgt, unlike_tensor_src_mask, unlike_tensor_tgt_mask)

            # ori_latent, like_latent, unlike_latent = ori_latent.detach(), like_latent.detach(), unlike_latent.detach()

            loss, choice = get_contrastive_loss(siamese_net,
                                                predictor_net,
                                                criterion,
                                                ori_latent,
                                                like_latent,
                                                unlike_latent,
                                                args)

            choice = np.array(choice).flatten()
            epoch_loss += loss.item()

            loss = loss * min((1+epoch)/args.anneal_epoch, 0.5)

            optimizer.zero_grad()
            if args.together:
                ae_optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.together:
                ae_optimizer.step()

            cur_num += len(choice)
            cur_acc += sum(np.array(choice) == 0)

            if it % 200 == 0:
                pbar.set_description(
                    'batch {}, ae_model loss={:.3f}, contrastive loss={:.3f}, acc={:.3f}'.format(it, loss_rec, loss,
                                                                                     cur_acc / cur_num))
                # print_and_log(
                # 'epoch {}, batch {}, loss={}, acc={}'.format(epoch, it, loss, cur_acc/cur_num))

        # test_siamese(siamese_net, predictor_net, ae_model, args, epoch)
        # evaluate_attack_dataset_acc(siamese_net, predictor_net, ae_model, args)
        print_and_log(
            f'Epoch ae_model={epoch_rec_loss} Epoch Loss={epoch_loss}')
        torch.save(ae_model.state_dict(), args.current_save_path +
                   'together_ae_model_params.pkl')

        save_siamese(siamese_net, predictor_net, args)

        siamese_net.train()
        predictor_net.train()


def train_siamese_by_label(siamese_net: SiameseNet,
                           predictor_net: Predictor,
                           ae_model: EncoderDecoder,
                           args,
                           epoch_num: int = 1,
                           label_to_train: int = 0):
    r"""
    Train the Siamese Net only using sentences with label 0
    """
    label_data_loader = create_train_data_loader(args)[label_to_train]

    ae_model.eval()
    siamese_net.train()
    predictor_net.train()

    if args.simplest_siamese:
        optimizer = optim.Adam(siamese_net.parameters(),
                               lr=args.train_siamese_lr)
    else:
        params = list(siamese_net.parameters()) + \
            list(predictor_net.parameters())
        optimizer = optim.Adam(params, lr=args.train_siamese_lr)

    criterion = get_cuda(nn.CosineSimilarity())

    for epoch in range(epoch_num):
        print_and_log(
            Headline('Traing on epoch.{} on Label.{}'.format(epoch, label_to_train)))
        for it in range(label_data_loader.num_batch):
            batch_sentences, tensor_labels, \
                tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
                tensor_tgt_mask, tensor_ntokens = label_data_loader.next_batch()

            latent, out = ae_model.forward(
                tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

            latent_reverse = shuffle_tensor(latent)

            optimizer.zero_grad()

            z1, z2 = siamese_net(latent), siamese_net(latent_reverse)
            p1, p2 = predictor_net(z1), predictor_net(z2)
            z1, z2 = z1.detach(), z2.detach()
            loss = - (criterion(p1, z2) + criterion(p2, z1)).mean() / 2

            loss.backward()
            optimizer.step()

            if it % 200 == 0:
                logging.info(
                    'Training on {}:epoch {}, batch {}, loss={}'.format(label_to_train, epoch, it, loss))

        test_siamese(siamese_net, predictor_net, ae_model, args, epoch)
        save_siamese(siamese_net, predictor_net, args)
        siamese_net.train()
        predictor_net.train()


def print_latent(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args):
    ori_attack_data_loader, like_attack_data_loader, unlike_attack_data_loader = create_train_data_loader_contrastive(
        args, 1, 'attack', pos_num=1, neg_num=1)

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()

    label_data = []
    h_data = []
    z_data = []

    for it in range(ori_attack_data_loader.num_batch):
        ori_src, ori_tgt, ori_src_mask, ori_tgt_mask = ori_attack_data_loader.next_batch()
        ori_latent, _ = ae_model.forward(
            ori_src, ori_tgt, ori_src_mask, ori_tgt_mask)
        ori_z = siamese_net(ori_latent)

        cur_label = 0 if it < 500 else 1
        label_data.append(cur_label)
        add_tensor_to_list(h_data, ori_latent)
        add_tensor_to_list(z_data, ori_z)

    h_data = np.concatenate(h_data)
    np.save('h_data.npy', h_data)

    z_data = np.concatenate(z_data)
    np.save('z_data.npy', z_data)

    label_data = np.array(label_data)
    np.save('label_data.npy', label_data)


def create_test_data_loader(args, batch_size):
    r"""
    Return zero_data_loader, one_data_loader
    """
    zero_data_loader = non_pair_data_loader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    one_data_loader = non_pair_data_loader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )

    zero_file_list = [args.data_path + 'sentiment.test.0']
    zero_file_list = [args.data_path + 'similar_ref0']
    zero_label_list = [[0]]

    one_file_list = [args.data_path + 'sentiment.test.1']
    one_file_list = [args.data_path + 'similar1_0']
    one_label_list = [[1]]

    zero_data_loader.create_batches(
        zero_file_list, zero_label_list, if_shuffle=False
    )
    one_data_loader.create_batches(
        one_file_list, one_label_list, if_shuffle=False
    )

    return zero_data_loader, one_data_loader


def create_attack_data_loader(args, batch_size):
    r"""
    Return zero_data_loader, one_data_loader
    """
    attack_data_loader = AttackDataLoader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )

    attack_files = [args.data_path + 'similar_ref0.json',
                    args.data_path + 'similar_ref1.json']

    if args.diff:
        attack_files = [args.data_path + 'diff_ref0.json',
                        args.data_path + 'diff_ref1.json']

    if args.reverse_order:
        attack_files.reverse()

    attack_data_loader.create_batches(attack_files)

    return attack_data_loader


def add_tensor_to_list(data: List, t: Tensor):
    c = copy.deepcopy
    data.append(t.clone().detach().cpu().numpy())


def evaluate_attack_dataset_acc(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args):
    # attack_data_loader = create_attack_data_loader(args, 1)
    ori_data_loader, sim_data_loader, ref_data_loader = create_train_data_loader_contrastive(
        args, 1, 'attack')

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()
    criterion = get_cuda(nn.NLLLoss())

    # latent_vdata = []
    # z_vdata = []

    tot, acc = 0, 0
    with torch.no_grad():
        for it in range(ori_data_loader.num_batch):
            ori_src, ori_tgt, ori_src_mask, ori_tgt_mask = ori_data_loader.next_batch()
            sim_src, sim_tgt, sim_src_mask, sim_tgt_mask = sim_data_loader.next_batch()
            ref_src, ref_tgt, ref_src_mask, ref_tgt_mask = ref_data_loader.next_batch()

            ori_latent, _ = ae_model.forward(
                ori_src, ori_tgt, ori_src_mask, ori_tgt_mask)
            sim_latent, _ = ae_model.forward(sim_src.squeeze(1), sim_tgt.squeeze(
                1), sim_src_mask.squeeze(1), sim_tgt_mask.squeeze(1))
            ref_latent, _ = ae_model.forward(ref_src.squeeze(1), ref_tgt.squeeze(
                1), ref_src_mask.squeeze(1), ref_tgt_mask.squeeze(1))

            ori_latent, sim_latent, ref_latent = ori_latent.detach(
            ), sim_latent.detach(), ref_latent.detach()

            loss, choice = get_contrastive_loss(siamese_net,
                                                predictor_net,
                                                criterion,
                                                ori_latent,
                                                sim_latent.unsqueeze(1),
                                                ref_latent.unsqueeze(1),
                                                args)

            # if (args.use_min and min(choice) == 0): acc += 1
            # if (not args.use_min and max(choice) == 0): acc += 1
            # tot += 1

            tot += len(choice)
            acc += sum(choice == 0)

    print_and_log(
        Headline('Evaluating Acc of Attack pairs: {}'.format(acc / tot)))

    # latent_vdata = np.concatenate(latent_vdata)
    # np.save('h.npy', latent_vdata)

    # z_vdata = np.concatenate(z_vdata)
    # np.save('z.npy', z_vdata)


def attack_one(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, victim: torch.Tensor,
               reference: torch.Tensor, args):
    r"""
    victim = [hidden_dim]
    reference = [ref_num, hidden_dim]
    """
    criterion = get_cuda(nn.CosineSimilarity())
    victim.requires_grad = True
    reference.requires_grad = False
    ref_num = reference.size(0)
    generator_text: str

    victim_optimizer = getattr(optim, args.victim_optimizer)([
        victim], lr=args.victim_lr)

    print_and_log(Headline('', note_token='-'))

    for it in range(args.attack_max_iter):
        z1 = siamese_net(victim)
        z2_list = siamese_net(reference)

        losses = []

        if args.simplest_siamese:
            for i in range(ref_num):
                cur_loss = criterion(z1, z2_list[i].unsqueeze(0))
                losses.append(cur_loss)
        else:
            p1 = predictor_net(z1)
            for i in range(ref_num):
                cur_loss = criterion(p1, z2_list[i].unsqueeze(0))
                losses.append(cur_loss)

        loss = losses[0]
        for i in range(1, ref_num):
            loss += losses[i]

        loss = loss / float(ref_num)

        victim_optimizer.zero_grad()
        loss = (-loss)
        loss.backward()
        victim_optimizer.step()

        generator_id = ae_model.greedy_decode(
            victim, max_len=args.max_sequence_length, start_id=args.id_bos)
        generator_text = id2text_sentence(generator_id[0], args.id_to_word)

        print_and_log('iteration.{}, {}'.format(it, generator_text))
    return generator_text


def add_to_output(sent: str, output_path: str):
    with open(output_path, 'a') as f:
        f.write(sent + '\n')
    return


def attack_one_pair(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, victim: torch.Tensor,
                    refs: torch.Tensor, args) -> None:
    # FIXME:
    c = copy.deepcopy
    ref_num = refs.size(0)
    # for i in range(ref_num):
    # attack_one(siamese_net, predictor_net, ae_model,
    #    c(victim), c(refs[i].unsqueeze(0)), args)

    attack_one(siamese_net, predictor_net, ae_model, c(victim), c(refs), args)

    # add_to_output(generate_021, os.path.join(args.output_path,
    #  '021-' + args.victim_optimizer + '+' + str(args.victim_lr)))
    # add_to_output(generate_120, os.path.join(args.output_path,
    #                                          '120-' + args.victim_optimizer + '+' + str(args.victim_lr)))


def siamese_attack(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args):
    attack_data_loader = create_attack_data_loader(args, 1)

    gold_ans = load_human_answer(args.data_path)

    print_and_log(Headline('Start Attack via Siamese Net'))
    print_and_log(Headline('Optimizer {}, lr {}, Iter {}'.format(
        args.victim_optimizer, args.victim_lr, args.attack_max_iter)))
    if args.simplest_siamese:
        print_and_log(
            Headline('Attack with Simplest Siamest Net', note_token='+'))
    else:
        print_and_log(
            Headline('Attack with enhance Siamese Net(with Predictor)', note_token='+'))

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()

    for it in range(attack_data_loader.num_batch):

        tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask = attack_data_loader.next_batch()

        print_and_log(Headline('Sentence Pair.{}'.format(it)))
        print_and_log('Src: ' + id2text_sentence(
            tensor_src[0], args.id_to_word))

        latent, _ = ae_model.forward(
            tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

        generator_text = ae_model.greedy_decode(latent,
                                                max_len=args.max_sequence_length,
                                                start_id=args.id_bos)

        print_and_log('Rec: ' + id2text_sentence(
            generator_text[0], args.id_to_word))
        for i in range(1, attack_data_loader.ref_num + 1):
            print_and_log('Ref{}: '.format(i) + id2text_sentence(
                generator_text[i], args.id_to_word))

        latent = latent.detach()
        attack_one_pair(siamese_net, predictor_net, ae_model,
                        latent[0].unsqueeze(0), latent[1:], args)

        # TODO: This need to be deleted
        # if it > 200:
        #     break


def print_latent_sent(t: torch.Tensor, ae_model: EncoderDecoder, args) -> None:
    dim = t.dim()
    if dim == 1:
        t = t.unsqueeze(0)
    g = ae_model.greedy_decode(t, max_len=20, start_id=args.id_bos)
    for i in range(t.size(0)):
        print(id2text_sentence(g[i], args.id_to_word))


def save_siamese(siamese_net: SiameseNet, predictor_net: Predictor, args):
    print_and_log('Saving Siamese Net to {}'.format(args.current_save_path))

    torch.save(siamese_net.state_dict(), args.current_save_path +
               'siamese_model_params.pkl')
    torch.save(predictor_net.state_dict(),
               args.current_save_path + 'predictor_net_params.pkl')


def load_siamese(siamese_net: SiameseNet, predictor_net: Predictor, args):
    logging.info('Loading Siamese network from {}'.format(
        args.current_save_path))
    print('Loading Siamese Net from {}'.format(args.current_save_path))

    siamese_net_path = args.current_save_path + 'siamese_model_params.pkl'
    predictor_net_path = args.current_save_path + 'predictor_net_params.pkl'

    if not os.path.exists(siamese_net_path):
        print('Siamese Net Path not Exist')
        exit(0)
    if not os.path.exists(predictor_net_path):
        print('Predictor Net Path Not Exist')
        exit(0)

    if use_cuda:
        siamese_net.load_state_dict(torch.load(siamese_net_path))
        predictor_net.load_state_dict(torch.load(predictor_net_path))
    else:
        siamese_net.load_state_dict(torch.load(
            siamese_net_path, map_location='cpu'))
        predictor_net.load_state_dict(torch.load(
            predictor_net_path, map_location='cpu'))


def print_and_log(sent: str):
    print(sent)
    logging.info(sent)


def initialize_logging(args):
    now_time = time.strftime("[%Y-%m-%d %H%M%S]", time.localtime())
    filename = args.victim_optimizer + '+' + \
        str(args.victim_lr) + '+iter' + str(args.attack_max_iter) + now_time
    path_dir = './loggings'
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    file_path = os.path.join(path_dir, filename)
    print('Saving Siamese Log to {}'.format(file_path))
    logging.basicConfig(filename=file_path, level=logging.DEBUG,
                        format="%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d - %(message)s")


def initialize_output(args):
    args.output_path = './outputs'
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)


###########################################
# VAE Section
###########################################


def print_vae_latent(enc_output, hidden, ae_model, args, if_return=False):
    # dim = hidden.dim()
    # if dim == 1:
    #     hidden = hidden.unsqueeze(0)

    dec_input = [args.id_eos]
    dec_input = get_cuda(torch.tensor(dec_input, dtype=torch.long))
    outputs = []
    for it in range(20):
        dec_output, hidden = ae_model.decoder(dec_input, hidden, enc_output)
        top1 = dec_output.argmax(1)
        outputs.append(top1.item())
        dec_input = top1
        if outputs[-1] == args.id_eos:
            break
    if if_return:
        return id2text_sentence(outputs, args.id_to_word)
    print(id2text_sentence(outputs, args.id_to_word))


def create_vae_attack_data_loader(args, batch_size=1):
    dataset = create_data_loader([args.data_path + 'sentiment.test.0', args.data_path + 'sentiment.test.1'], [0, 1],
                                 args.batch_size, args.id_bos, args.id_eos, args.id_unk, args.max_sequence_length,
                                 args.vocab_size, if_shuffle=False)
    attack_data = TensorDataset(*dataset)
    attack_data_loader = DataLoader(attack_data, batch_size=batch_size)

    return attack_data_loader


def vae_siamese_attack(siamese_net: SiameseNet, predictor_net: Predictor, ae_model, args):
    attack_data_loader = create_attack_data_loader(args, 1)

    gold_ans = load_human_answer(args.data_path)

    print_and_log(Headline('Start Attack via Siamese Net'))
    print_and_log(Headline('Optimizer {}, lr {}, Iter {}'.format(
        args.victim_optimizer, args.victim_lr, args.attack_max_iter)))
    if args.simplest_siamese:
        print_and_log(
            Headline('Attack with Simplest Siamest Net', note_token='+'))
    else:
        print_and_log(
            Headline('Attack with enhance Siamese Net(with Predictor)', note_token='+'))

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()

    for it in range(attack_data_loader.num_batch):

        tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask = attack_data_loader.next_batch()
        tensor_src_len = (tensor_src != 0).sum(-1)

        print_and_log(Headline('Sentence Pair.{}'.format(it)))
        print_and_log('Src: ' + id2text_sentence(
            tensor_src[0], args.id_to_word))

        enc_output, hidden = ae_model.encoder_only(tensor_src, tensor_src_len)
        hidden, _, _ = ae_model.re_hidden(hidden)

        print_and_log(
            'Rec: ' + print_vae_latent(enc_output[:, :1, :], hidden[:1], ae_model, args, if_return=True))

        for i in range(1, hidden.size(0)):
            print_and_log(f'Ref{i}: ' + print_vae_latent(enc_output[:, i:i + 1, :],
                                                         hidden[i:i + 1], ae_model, args, if_return=True))
        hidden, enc_output = hidden.detach(), enc_output.detach()
        vae_attack_one(siamese_net, predictor_net, ae_model,
                       hidden[:1], hidden[1:], enc_output[:, :1, :], args)


def test_vae_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model, args, epoch_num: int = 0):
    dataset = create_data_loader([args.data_path + 'sentiment.test.0', args.data_path + 'sentiment.test.1'], [0, 1],
                                 args.batch_size, args.id_bos, args.id_eos, args.id_unk, args.max_sequence_length,
                                 args.vocab_size)
    test_data = TensorDataset(*dataset)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()
    criterion = get_cuda(nn.CosineSimilarity())

    latent_vdata = []
    z_vdata = []

    with torch.no_grad():
        Loss_Accumulator = Accumulator('Loss')
        Acc_p_Accumulator = Accumulator('Acc Positive')
        Acc_n_Accumulator = Accumulator('Acc Negative')
        print(Headline('Test in epoch {}'.format(epoch_num)))

        for it, batch in enumerate(test_data_loader):
            zero_tensor_labels, zero_tensor_src, zero_tensor_tgt, zero_tensor_tgt_y, zero_tensor_src_len, \
                one_tensor_labels, one_tensor_src, one_tensor_tgt, one_tensor_tgt_y, one_tensor_src_len = batch

            zero_enc_output, zero_hidden = ae_model.encoder_only(
                zero_tensor_src, zero_tensor_src_len)
            one_enc_output, one_hidden = ae_model.encoder_only(
                one_tensor_src, one_tensor_src_len)

            zero_hidden, _, _ = ae_model.re_hidden(zero_hidden)
            one_hidden, _, _ = ae_model.re_hidden(one_hidden)

            zero_hidden_shuffle = shuffle_tensor(zero_hidden)
            one_hidden_shuffle = shuffle_tensor(one_hidden)

            add_tensor_to_list(latent_vdata, zero_hidden)

            r"""
            Positive Samples
            """
            new_latent = torch.cat([zero_hidden, one_hidden])
            new_reverse_latent = torch.cat(
                [zero_hidden_shuffle, one_hidden_shuffle])

            if args.simplest_siamese:
                z1, z2 = siamese_net(new_latent), siamese_net(
                    new_reverse_latent)
                loss = - criterion(z2, z1).mean()
                acc = (criterion(z1, z2) > 0).sum()
            else:
                z1, z2 = siamese_net(new_latent), siamese_net(
                    new_reverse_latent)
                p1, p2 = predictor_net(z1), predictor_net(z2)
                loss = - (criterion(p1, z2.detach()) +
                          criterion(p2, z1.detach())).mean() / 2
                acc = ((criterion(z1, p2) > 0).sum() +
                       (criterion(z2, p1) > 0).sum()).float() / 2

            Loss_Accumulator.inc(loss.item())
            Acc_p_Accumulator.inc(acc.item(), new_latent.size(0))

            r"""
            Negative Samples
            """
            new_latent = torch.cat([zero_hidden, zero_hidden_shuffle])
            new_reverse_latent = torch.cat([one_hidden, one_hidden_shuffle])

            z1, z2 = siamese_net(new_latent), siamese_net(new_reverse_latent)
            if args.simplest_siamese:
                acc = (criterion(z1, z2) < 0).sum()
            else:
                p1, p2 = predictor_net(z1), predictor_net(z2)
                c1 = (criterion(z1, p2))
                c2 = (criterion(z2, p1))
                acc = ((c1 < 0).sum() + (c2 < 0).sum()).float() / 2

            Acc_p_Accumulator.inc(acc.item(), new_latent.size(0))

        latent_vdata = np.concatenate(latent_vdata)
        np.save('h.npy', latent_vdata)

        print_and_log(str(Loss_Accumulator))
        print_and_log(str(Acc_p_Accumulator))
        print_and_log(str(Acc_n_Accumulator))


def train_vae_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder,
                      args,
                      train_data_loader,
                      epoch_num: int = 1):
    r"""
    Training Siamese Net
    """

    ae_model.eval()
    siamese_net.train()
    predictor_net.train()

    optimizer = None

    if args.simplest_siamese:
        optimizer = optim.Adam(siamese_net.parameters(),
                               lr=args.train_siamese_lr)
    else:
        params = list(siamese_net.parameters()) + \
            list(predictor_net.parameters())
        optimizer = optim.Adam(params, lr=args.train_siamese_lr)

    criterion = get_cuda(nn.CosineSimilarity())

    print_and_log(Headline('Training Siamese by lr={}'.format(
        args.train_siamese_lr), note_token='+'))

    for epoch in range(epoch_num):
        print_and_log(
            Headline('Traing VAE Siamese for epoch.{}'.format(epoch)))

        pbar = tqdm(enumerate(train_data_loader))
        epoch_loss = 0

        for it, batch in pbar:
            zero_tensor_labels, zero_tensor_src, zero_tensor_tgt, zero_tensor_src_mask, zero_tensor_tgt_mask, \
                one_tensor_labels, one_tensor_src, one_tensor_tgt, one_tensor_src_mask, one_tensor_tgt_mask = batch

            zero_latent, _ = ae_model.forward(
                zero_tensor_src, zero_tensor_tgt, zero_tensor_src_mask, zero_tensor_tgt_mask)
            one_latent, _ = ae_model.forward(
                one_tensor_src, one_tensor_tgt, one_tensor_src_mask, one_tensor_tgt_mask)

            latent1, latent2, targets = None, None, None
            zero_latent_shuffle = shuffle_tensor(zero_latent)
            one_latent_shuffle = shuffle_tensor(one_latent)

            if args.negative_samples:
                latent1 = torch.cat(
                    [zero_latent, one_latent, zero_latent, one_latent_shuffle])
                latent2 = torch.cat(
                    [zero_latent_shuffle, one_latent_shuffle, one_latent, zero_latent_shuffle])
                targets = get_cuda(torch.tensor(
                    [1] * (zero_latent.size(0) * 2) + [-1] * (one_latent.size(0) * 2)).float())
            else:
                latent1 = torch.cat([zero_latent, one_latent])
                latent2 = torch.cat([zero_latent_shuffle, one_latent_shuffle])
                targets = get_cuda(torch.tensor(
                    [1] * (zero_latent.size(0) * 2)).float())

            z1, z2 = siamese_net(latent1), siamese_net(latent2)

            if args.simplest_siamese:
                loss = - (criterion(z2, z1) * targets).mean()
            else:
                p1, p2 = predictor_net(z1), predictor_net(z2)
                z1, z2 = z1.detach(), z2.detach()
                loss1 = (targets * criterion(p1, z2)).mean()
                loss2 = (targets * criterion(p2, z1)).mean()
                loss = - (loss1 + loss2) / 2

            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if it % 200 == 0:
                pbar.set_description(
                    'epoch {}, batch {:.3f}, loss={:.3f}'.format(epoch, it, loss))

        # test_vae_siamese(siamese_net, predictor_net, ae_model, args, epoch)
        save_siamese(siamese_net, predictor_net, args)

        siamese_net.train()
        predictor_net.train()


def vae_attack_one(siamese_net: SiameseNet, predictor_net: Predictor, ae_model, victim: torch.Tensor,
                   reference: torch.Tensor, enc_output, args):
    r"""
    victim = [hidden_dim]
    reference = [ref_num, hidden_dim]
    """
    criterion = get_cuda(nn.CosineSimilarity())
    victim.requires_grad = True
    reference.requires_grad = False
    ref_num = reference.size(0)
    generator_text: str

    victim_optimizer = getattr(optim, args.victim_optimizer)([
        victim], lr=args.victim_lr)

    print_and_log(Headline('', note_token='-'))

    for it in range(args.attack_max_iter):
        z1 = siamese_net(victim)
        z2_list = siamese_net(reference)

        losses = []

        if args.simplest_siamese:
            for i in range(ref_num):
                cur_loss = criterion(z1, z2_list[i].unsqueeze(0))
                losses.append(cur_loss)
        else:
            p1 = predictor_net(z1)
            for i in range(ref_num):
                cur_loss = criterion(p1, z2_list[i].unsqueeze(0))
                losses.append(cur_loss)

        loss = losses[0]
        for i in range(1, ref_num):
            loss += losses[i]

        loss = loss / float(ref_num)

        victim_optimizer.zero_grad()
        loss = (-loss)
        loss.backward()
        victim_optimizer.step()

        generator_text = print_vae_latent(
            enc_output, victim, ae_model, args, if_return=True)

        print_and_log('iteration.{}, {}'.format(it, generator_text))
    return generator_text


if __name__ == '__main__':
    initialize_logging()
    logging.info('initializing logging successfully')
    ae_model = get_cuda(make_model())
