from argparse import Namespace
from typing import List
# from vae import VAE
import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from model import EncoderDecoder, make_model, Classifier, NoamOpt, LabelSmoothing, fgim_attack
from data import AttackDataLoader, create_data_loader, prepare_data, non_pair_data_loader, get_cuda, pad_batch_seuqences, id2text_sentence,\
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

        self.fc1 = nn.Linear(latent_size, latent_size*2)
        self.norm1 = nn.BatchNorm1d(latent_size*2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(latent_size * 2, latent_size * 2)
        self.norm2 = nn.BatchNorm1d(latent_size * 2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(latent_size * 2, latent_size)
        self.norm3 = nn.BatchNorm1d(latent_size, affine=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        return self.norm3(x)


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
        Acc_Accumulator = Accumulator('Accuracy')

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
            # if args.simplest_siamese:
            #     z1, z2 = siamese_net(new_latent), siamese_net(
            #         new_reverse_latent)
            #     loss = - criterion(z2, z1).mean()
            #     acc = (criterion(z1, z2) > 0).sum()
            # else:
            #     z1, z2 = siamese_net(new_latent), siamese_net(
            #         new_reverse_latent)
            #     p1, p2 = predictor_net(z1), predictor_net(z2)
            #     loss = - (criterion(p1, z2.detach()) +
            #               criterion(p2, z1.detach())).mean() / 2
            #     acc = ((criterion(z1, p2) > 0).sum() +
            #            (criterion(z2, p1) > 0).sum()).float() / 2

            # Loss_Accumulator.inc(loss.item())
            # Acc_Accumulator.inc(acc.item(), new_latent.size(0))

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
                # acc = ((criterion(z1, p2) < 0).sum() +
                #        (criterion(z2, p1) < 0).sum()).float() / 2

            Acc_Accumulator.inc(acc.item(), new_latent.size(0))

            # We cannot run so many test cases on CPU
            if it > 100 and not use_cuda:
                break

        print_and_log(str(Acc_Accumulator))
        print_and_log(str(Loss_Accumulator))


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
            [1]*2*zero_latent.size(0) + [-1]*2*zero_latent.size(0)).float())
    else:
        latent1 = torch.cat([zero_latent, one_latent])
        latent2 = torch.cat([zero_latent_shuffle, one_latent_shuffle])
        targets = get_cuda(torch.tensor([1]*latent1.size(0)).float())

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


def train_ae_model(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args, epoch_num: int = 1):
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
            anneal_ratio = min(1.0, epoch/args.anneal_epoch)

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


def train_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args, epoch_num: int = 1):
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

    args.diff = 1
    if args.diff:
        attack_files = [args.data_path + 'diff_ref0.json',
                        args.data_path + 'diff_ref1.json']

    if args.reverse_order:
        attack_files.reverse()

    attack_data_loader.create_batches(attack_files)

    return attack_data_loader


def add_tensor_to_list(data: List, t: Tensor):
    c = copy.deepcopy
    data.append(t.detach().cpu().numpy())


def evaluate_attack_dataset_acc(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, args):
    attack_data_loader = create_attack_data_loader(args, 1)
    Acc_accumulator = Accumulator('Accuracy')
    ref_num = attack_data_loader.ref_num

    siamese_net.eval()
    predictor_net.eval()
    ae_model.eval()
    criterion = get_cuda(nn.CosineSimilarity())

    latent_vdata = []
    z_vdata = []

    with torch.no_grad():

        for it in range(attack_data_loader.num_batch):

            tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask = attack_data_loader.next_batch()

            latent, _ = ae_model.forward(
                tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
            latent = latent.detach()

            victim = latent[0].unsqueeze(0)
            references = latent[1:]

            add_tensor_to_list(latent_vdata, victim)

            z1, z2 = siamese_net(victim), siamese_net(references)
            p1 = predictor_net(z1)

            add_tensor_to_list(z_vdata, z1)

            for i in range(ref_num):
                cur_sim = criterion(p1, z2[i].unsqueeze(0))
                cur_acc = (cur_sim < 0).sum().item()
                Acc_accumulator.inc(cur_acc)

    print_and_log(
        Headline('Evaluating Acc of Attack pairs: {}'.format(Acc_accumulator.value)))

    latent_vdata = np.concatenate(latent_vdata)
    np.save('h.npy', latent_vdata)

    z_vdata = np.concatenate(z_vdata)
    np.save('z.npy', z_vdata)


def attack_one(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, victim: torch.Tensor, reference: torch.Tensor, args):
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


def attack_one_pair(siamese_net: SiameseNet, predictor_net: Predictor, ae_model: EncoderDecoder, victim: torch.Tensor, refs: torch.Tensor, args) -> None:
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
                                 args.batch_size, args.id_bos, args.id_eos, args.id_unk, args.max_sequence_length, args.vocab_size, if_shuffle=False)
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
            print_and_log(f'Ref{i}: ' + print_vae_latent(enc_output[:, i:i+1, :],
                                                         hidden[i:i+1], ae_model, args, if_return=True))
        hidden, enc_output = hidden.detach(), enc_output.detach()
        vae_attack_one(siamese_net, predictor_net, ae_model,
                       hidden[:1], hidden[1:], enc_output[:, :1, :], args)


def test_vae_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model, args, epoch_num: int = 0):

    dataset = create_data_loader([args.data_path + 'sentiment.test.0', args.data_path + 'sentiment.test.1'], [0, 1],
                                 args.batch_size, args.id_bos, args.id_eos, args.id_unk, args.max_sequence_length, args.vocab_size)
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
            zero_tensor_labels, zero_tensor_src, zero_tensor_tgt, zero_tensor_tgt_y, zero_tensor_src_len,\
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


def train_vae_siamese(siamese_net: SiameseNet, predictor_net: Predictor, ae_model, args, train_data_loader, epoch_num: int = 1):
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
        for it, batch in tqdm(enumerate(train_data_loader)):
            zero_tensor_labels, zero_tensor_src, zero_tensor_tgt, zero_tensor_tgt_y, zero_tensor_src_len,\
                one_tensor_labels, one_tensor_src, one_tensor_tgt, one_tensor_tgt_y, one_tensor_src_len = batch

            zero_enc_output, zero_hidden = ae_model.encoder_only(
                zero_tensor_src, zero_tensor_src_len)
            one_enc_output, one_hidden = ae_model.encoder_only(
                one_tensor_src, one_tensor_src_len)

            # print_vae_latent(zero_enc_output[:, :1, :], zero_hidden[:1], ae_model, args)

            zero_latent, _, _ = ae_model.re_hidden(zero_hidden)
            one_latent, _, _ = ae_model.re_hidden(one_hidden)
            r"""
            latent = [128, 256]
            """

            latent1, latent2, targets = None, None, None
            zero_latent_shuffle = shuffle_tensor(zero_latent)
            one_latent_shuffle = shuffle_tensor(one_latent)

            if args.negative_samples:
                latent1 = torch.cat(
                    [zero_latent, one_latent, zero_latent, one_latent_shuffle])
                latent2 = torch.cat(
                    [zero_latent_shuffle, one_latent_shuffle, one_latent, zero_latent_shuffle])
                targets = get_cuda(torch.tensor(
                    [1]*(zero_latent.size(0) * 2) + [-1] * (one_latent.size(0) * 2)).float())
            else:
                latent1 = torch.cat([zero_latent, one_latent])
                latent2 = torch.cat([zero_latent_shuffle, one_latent_shuffle])
                targets = get_cuda(torch.tensor(
                    [1]*(zero_latent.size(0) * 2)).float())

            optimizer.zero_grad()

            z1, z2 = siamese_net(latent1), siamese_net(latent2)

            if args.simplest_siamese:
                loss = - (criterion(z2, z1) * targets).mean()
            else:
                p1, p2 = predictor_net(z1), predictor_net(z2)
                z1, z2 = z1.detach(), z2.detach()
                loss1 = (targets * criterion(p1, z2)).mean()
                loss2 = (targets * criterion(p2, z1)).mean()
                loss = - (loss1 + loss2) / 2

            loss.backward()
            optimizer.step()

            if it % 200 == 0:
                logging.info(
                    'epoch {}, batch {}, loss={}'.format(epoch, it, loss))

        test_vae_siamese(siamese_net, predictor_net, ae_model, args, epoch)
        save_siamese(siamese_net, predictor_net, args)
        siamese_net.train()
        predictor_net.train()


def vae_attack_one(siamese_net: SiameseNet, predictor_net: Predictor, ae_model, victim: torch.Tensor, reference: torch.Tensor, enc_output, args):
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
