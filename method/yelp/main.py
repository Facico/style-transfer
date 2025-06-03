from siamese import Predictor, SiameseNet
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler
from vae import make_VAE_model, train_vae
# from siamese import Predictor, SiameseNet, evaluate_attack_dataset_acc, initialize_output, load_siamese, print_and_log, save_siamese, siamese_attack, test_siamese, test_vae_siamese, train_ae_model, train_siamese_by_label, train_vae_siamese, vae_siamese_attack
import time
import argparse
import math
import os
import torch
import torch.nn as nn
from torch import optim
import numpy
import matplotlib
from matplotlib import pyplot as plt

from model import EncoderDecoder, make_model, Classifier, NoamOpt, LabelSmoothing, fgim_attack
from data import create_data_loader, prepare_data, non_pair_data_loader, get_cuda, pad_batch_seuqences, id2text_sentence,\
    to_var, calc_bleu, load_human_answer, use_cuda
from siamese_contrastive import attack_contrastive, initialize_output, train_ae_model_constraint, train_together, \
    print_and_log, print_latent, save_siamese, train_siamese, initialize_logging, load_siamese, \
    train_contrastive, evaluate_attack_dataset_acc, Predictor, SiameseNet, initialize_logging,train_ae_model, train_vae_siamese

my_version = 1.242
# if use_cuda:
#     torch.cuda.set_device(3)

# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

######################################################################################
#  Environmental parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--id_pad', type=int, default=0, help='')
parser.add_argument('--id_unk', type=int, default=1, help='')
parser.add_argument('--id_bos', type=int, default=2, help='')
parser.add_argument('--id_eos', type=int, default=3, help='')

######################################################################################
#  File parameters
######################################################################################
parser.add_argument('--task', type=str, default='yelp',
                    help='Specify datasets.')
parser.add_argument('--word_to_id_file', type=str, default='', help='')
parser.add_argument('--data_path', type=str, default='', help='')

######################################################################################
#  VAE Parameter
######################################################################################
parser.add_argument('--encoder_emb_dim', type=int, default=256, help='')
parser.add_argument('--decoder_emb_dim', type=int, default=256, help='')
parser.add_argument('--encoder_hidden_dim', type=int, default=256, help='')
parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='')
parser.add_argument('--encoder_dropout', type=float, default=0.5, help='')
parser.add_argument('--decoder_dropout', type=float, default=0.5, help='')

parser.add_argument('--AE_LATENT_NUM', type=int, default=80, help='')
parser.add_argument('--max_epoch', type=int, default=80, help='')


######################################################################################
#  Model parameters
######################################################################################
parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_sequence_length', type=int, default=60)
parser.add_argument('--num_layers_AE', type=int, default=2)
parser.add_argument('--transformer_model_size', type=int, default=256)
parser.add_argument('--transformer_ff_size', type=int, default=1024)

parser.add_argument('--latent_size', type=int, default=256)
parser.add_argument('--word_dropout', type=float, default=1.0)
parser.add_argument('--embedding_dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--label_size', type=int, default=1)

######################################################################################
#  Siamese Parameter
######################################################################################
parser.add_argument('--train_siamese_lr', type=float, default=3e-3)
parser.add_argument('--simplest_siamese', action='store_true',
                    help="use the simplest version of siamese network of the enhanced one(with predictor)")
parser.add_argument('--negative_samples', action='store_true')
parser.add_argument('--tfidf_train', action='store_true',
                    help='If use TF-IDF pairs to train Siamese')
parser.add_argument('--anneal_epoch', type=int, default=100)
parser.add_argument('--use_min', action='store_true',
                    help='use minimum of choice(for anyone right), default is use maximum(for all right)')
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--bata', type=float, default=0.1, help='Propotion of constraint v.s. reconstruction')
parser.add_argument('--together', action='store_true',
                    help='If use TF-IDF pairs to train Siamese')


######################################################################################
#  Siamese Attack Parameter
######################################################################################
parser.add_argument('--victim_optimizer', type=str,
                    default='Adam', help='Optimizer on latent')
parser.add_argument('--victim_lr', type=float, default=0.25)
parser.add_argument('--attack_max_iter', type=int, default=18)
parser.add_argument('--attack_max_change', type=int, default=7)
parser.add_argument('--reverse_order', action='store_true',
                    help='By default, first negative sentences will be attacked, then positive sentences be attacked. This option reverse the order')
parser.add_argument('--diff', action='store_true',
                    help='Attack with most similar or most difference, default is similar')
parser.add_argument('--just_use_one', action='store_true',
                    help='Attack with minimum loss of all positive, default is using all loss')
parser.add_argument('--attack', action='store_true',
                    help='default is train_contrastive') # FIXME: This need to be deleted later
parser.add_argument('--retext', action='store_true',
                    help='If do retext operation while attacking') # FIXME: This need to be deleted later
parser.add_argument('--not_nll', action='store_true',
                    help='If use nll loss when attacking') # FIXME: This need to be deleted later
parser.add_argument('--early_stop', action='store_true',
                    help='If stop attacking once the retext text is classified to the target domain')
parser.add_argument('--dance', action='store_true', help='Only attack once to get anti-dis-attack performance')
parser.add_argument('--pos_num', type=int, default=None)
parser.add_argument('--neg_num', type=int, default=None)

parser.add_argument('--pos_victim_lr', type=float, default=0.25)
parser.add_argument('--neg_victim_lr', type=float, default=0.25)

args = parser.parse_args()

# args.if_load_from_checkpoint = False
# args.checkpoint_name = "1627398507"
args.if_load_from_checkpoint = True
args.checkpoint_name = "origin"


######################################################################################
#  End of hyper parameters
######################################################################################


def add_log(ss):
    now_time = time.strftime("[%Y-%m-%d %H:%M:%S]: ", time.localtime())
    print(now_time + ss)
    with open(args.log_file, 'a') as f:
        f.write(now_time + str(ss) + '\n')
    return


def add_output(ss):
    with open(args.output_file, 'a') as f:
        f.write(str(ss) + '\n')
    return


def preparation():
    # set model save path
    if args.if_load_from_checkpoint:
        timestamp = args.checkpoint_name
    else:
        timestamp = str(int(time.time()))
    args.current_save_path = 'save/%s/' % timestamp
    args.log_file = args.current_save_path + \
        time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", time.localtime())
    args.output_file = args.current_save_path + \
        time.strftime("output_%Y_%m_%d_%H_%M_%S.txt", time.localtime())

    if os.path.exists(args.current_save_path):
        add_log("Load checkpoint model from Path: %s" % args.current_save_path)
    else:
        os.makedirs(args.current_save_path)
        add_log("Path: %s is created" % args.current_save_path)

    # set task type
    if args.task == 'yelp':
        if torch.cuda.is_available():   # Running on Colab
            args.data_path = '../../data/yelp/processed_files/'
        else:
            args.data_path = '../../data/yelp/processed_files/'
    elif args.task == 'amazon':
        if torch.cuda.is_available():
            args.data_path = '../../data/amazon/processed_files/'
        else:
            args.data_path = '../../data/amazon/processed_files/'
    elif args.task == 'imagecaption':
        pass
    else:
        raise TypeError('Wrong task type!')

    # args.data_path = '../../data/yelp/processed_files/'

    # prepare data
    args.id_to_word, args.vocab_size, \
        args.train_file_list, args.train_label_list = prepare_data(
            data_path=args.data_path, max_num=args.word_dict_max_num, task_type=args.task
        )
    word_to_id = {}
    for id, word in enumerate(args.id_to_word):
        word_to_id[word] = id
    args.word_to_id = word_to_id

    return
    

def train_dis(ae_model, dis_model):
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    train_data_loader.create_batches(args.train_file_list, args.train_label_list, if_shuffle=True)
    add_log("Start train process.")
    ae_model.eval()
    dis_model.train()

    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=0.0001)

    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    dis_criterion = nn.BCELoss(size_average=True)

    for epoch in range(10):
        print('-' * 94)
        epoch_start_time = time.time()
        for it in range(train_data_loader.num_batch):
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = train_data_loader.next_batch()

            # Forward pass
            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

            # Classifier
            dis_lop = dis_model.forward(latent.detach())

            loss_dis = dis_criterion(dis_lop, tensor_labels)

            dis_optimizer.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()

            if it % 200 == 0:
                add_log(
                    '| epoch {:3d} | {:5d}/{:5d} batches | dis loss {:5.4f} |'.format(
                        epoch, it, train_data_loader.num_batch, loss_dis))

        add_log(
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))
        # Save model
        torch.save(dis_model.state_dict(), args.current_save_path + 'dis_model_params.pkl')
    return


def train_iters(ae_model, dis_model):
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    train_data_loader.create_batches(args.train_file_list, args.train_label_list, if_shuffle=True)
    add_log("Start train process.")
    ae_model.train()
    dis_model.train()

    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=0.0001)

    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    dis_criterion = nn.BCELoss(size_average=True)

    for epoch in range(30):
        print('-' * 94)
        epoch_start_time = time.time()
        for it in range(train_data_loader.num_batch):
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = train_data_loader.next_batch()

            # Forward pass
            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

            # Loss calculation
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data

            # Classifier
            dis_lop = dis_model.forward(latent.clone())

            loss_dis = dis_criterion(dis_lop, tensor_labels)

            ae_optimizer.optimizer.zero_grad()
            dis_optimizer.zero_grad()

            (0.5*loss_rec + loss_dis).backward()

            ae_optimizer.step()
            dis_optimizer.step()

            if it % 200 == 0:
                add_log(
                    '| epoch {:3d} | {:5d}/{:5d} batches | dis loss {:5.4f} |'.format(
                        epoch, it, train_data_loader.num_batch, loss_dis))

                print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
                generator_text = ae_model.greedy_decode(latent,
                                                        max_len=args.max_sequence_length,
                                                        start_id=args.id_bos)
                print(id2text_sentence(generator_text[0], args.id_to_word))

        add_log(
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))
        # Save model
        torch.save(ae_model.state_dict(), args.current_save_path + 'ae_model_params.pkl')
        torch.save(dis_model.state_dict(), args.current_save_path + 'dis_model_params.pkl')
    return


def eval_iters(ae_model, dis_model):
    eval_data_loader = non_pair_data_loader(
        batch_size=1, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    eval_file_list = [
        args.data_path + 'sentiment.test.0',
        args.data_path + 'sentiment.test.1',
    ]
    eval_label_list = [
        [0],
        [1],
    ]
    eval_data_loader.create_batches(eval_file_list, eval_label_list, if_shuffle=False)
    gold_ans = load_human_answer(args.data_path)
    assert len(gold_ans) == eval_data_loader.num_batch


    add_log("Start eval process.")
    ae_model.eval()
    dis_model.eval()
    for it in range(eval_data_loader.num_batch):
        batch_sentences, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, tensor_ntokens = eval_data_loader.next_batch()

        print("------------%d------------" % it)
        print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
        print("origin_labels", tensor_labels)

        latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
        generator_text = ae_model.greedy_decode(latent,
                                                max_len=args.max_sequence_length,
                                                start_id=args.id_bos)
        print(id2text_sentence(generator_text[0], args.id_to_word))

        # Define target label
        target = get_cuda(torch.tensor([[1.0]], dtype=torch.float))
        if tensor_labels[0].item() > 0.5:
            target = get_cuda(torch.tensor([[0.0]], dtype=torch.float))
        print("target_labels", target)

        modify_text = fgim_attack(dis_model, latent, target, ae_model, args.max_sequence_length, args.id_bos,
                                        id2text_sentence, args.id_to_word, gold_ans[it], args)
        add_output(modify_text)
    return


def OMG(args):
    if_test = True
    dataset = None
    if if_test:
        dataset = create_data_loader([args.data_path + 'sentiment.test.0', args.data_path + 'sentiment.test.1'], [0, 1],
                                     args.batch_size, args.id_bos, args.id_eos, args.id_unk, args.max_sequence_length, args.vocab_size)
    else:
        dataset = create_data_loader([args.data_path + 'sentiment.train.0', args.data_path + 'sentiment.train.1'], [0, 1],
                                     args.batch_size, args.id_bos, args.id_eos, args.id_unk, args.max_sequence_length, args.vocab_size)

    train_data = TensorDataset(*dataset)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)

    siamese_net = get_cuda(SiameseNet(args.decoder_hidden_dim))
    predictor_net = get_cuda(Predictor(args.decoder_hidden_dim))

    # vae = make_VAE_model(args)
    vae = get_cuda(make_model(d_vocab=args.vocab_size,
                                   N=args.num_layers_AE,
                                   d_model=args.transformer_model_size,
                                   latent_size=args.latent_size,
                                   d_ff=args.transformer_ff_size,
                                   ))

    train_vae_siamese(siamese_net, predictor_net, vae, args, train_data_loader, epoch_num=10)

    load_siamese(siamese_net, predictor_net, args)

    if not use_cuda:
        vae.load_state_dict(torch.load(
            args.current_save_path+'vae_model.pkl', map_location='cpu'))
        print('--- VAE model loaded')
    else:
        vae.load_state_dict(torch.load(args.current_save_path+'vae_model.pkl'))
        print('--- VAE model loaded')

    # test_vae_siamese(siamese_net, predictor_net, vae, args)

    # train_vae_siamese(siamese_net, predictor_net, vae, args, train_data_loader, epoch_num=50)
    # vae_siamese_attack(siamese_net, predictor_net, vae, args)

    # train_vae(vae,  train_data_loader, args)


def print_basic_information():
    print_and_log('Path: {}'.format(os.path.abspath('.')))
    if args.if_load_from_checkpoint:
        print_and_log('--- Checkpoint Path: {}'.format(args.checkpoint_name))
        print_and_log('--- AE Model Name: {}'.format(args.ae_checkpoint))
    print_and_log('--- Attack: {}'.format(args.attack))
    print_and_log('--- pos_num: {}'.format(args.pos_num))
    print_and_log('--- neg_num: {}'.format(args.neg_num))
    print_and_log('--- pos_victim_lr: {}'.format(args.pos_victim_lr))
    print_and_log('--- neg_victim_lr: {}'.format(args.neg_victim_lr))
    print_and_log('--- not_nll: {}'.format(args.not_nll))
    print_and_log('--- simplest_siamese: {}'.format(args.simplest_siamese))
    print_and_log(str(args))

if __name__ == '__main__':
    initialize_logging(args)
    initialize_output(args)

    print('--- Running Siamese Style Transfer, Version{}'.format(my_version))
    print('--- Cuda: {}'.format(use_cuda))

    args.ae_checkpoint = 'constraint_ae_model0_params.pkl'
    # args.ae_checkpoint = 'ae_model_params.pkl'

    print_basic_information()
    preparation()

    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                   N=args.num_layers_AE,
                                   d_model=args.transformer_model_size,
                                   latent_size=args.latent_size,
                                   d_ff=args.transformer_ff_size,
                                   ))
    dis_model = get_cuda(Classifier(
        latent_size=args.latent_size, output_size=args.label_size))

    siamese_net = get_cuda(SiameseNet(args.latent_size))
    predictor_net = get_cuda(Predictor(args.latent_size))

    # OMG(args)

    if args.if_load_from_checkpoint:
        if use_cuda:
            # ae_model.load_state_dict(torch.load(
                # args.current_save_path + 'together_ae_model_params.pkl'))
            ae_model.load_state_dict(torch.load(
                args.current_save_path + args.ae_checkpoint, map_location='cuda:0'))
            dis_model.load_state_dict(torch.load(args.current_save_path + 'dis_model_params.pkl', map_location='cuda:0'))
            args.dis_model = dis_model
        else:
            # ae_model.load_state_dict(torch.load(
            #     args.current_save_path + 'new_ae_model_params.pkl', map_location='cpu'))
            ae_model.load_state_dict(torch.load(
            args.current_save_path + args.ae_checkpoint, map_location='cpu'))
            # dis_model.load_state_dict(torch.load(args.current_save_path + 'dis_model_params.pkl', map_location='cpu'))
        if args.attack:
            load_siamese(siamese_net, predictor_net, args)
    else:
        train_ae_model_constraint(ae_model, args)
        # train_together(siamese_net, predictor_net, ae_model, args, 250)

    # train_siamese_by_label(siamese_net, predictor_net, ae_model, args, 20, 0)

    # train_ae_model(siamese_net, predictor_net, ae_model, args, 200)
    # train_together(siamese_net, predictor_net, ae_model, args, 250)

    # train_ae_model_constraint(ae_model, args, 50)
    # train_dis(ae_model, dis_model)
    # eval_iters(ae_model, dis_model)

    # print_and_log('Done')
    # exit(0)

    if args.attack:
        attack_contrastive(siamese_net, predictor_net, ae_model, args)
    else:
        # load_siamese(siamese_net, predictor_net, args)
        train_contrastive(siamese_net, predictor_net, ae_model, args, 50)

    print('Done')
    exit(0)

    # load_siamese(siamese_net, predictor_net, args)
    # print_latent(siamese_net, predictor_net, ae_model, args)
    # evaluate_attack_dataset_acc(siamese_net, predictor_net, ae_model, args)
    # train_siamese(siamese_net, predictor_net, ae_model, args, 50)

    # siamese_attack(siamese_net, predictor_net, ae_model, args)

    # evaluate_attack_dataset_acc(siamese_net, predictor_net,
    # ae_model, args)
    # siamese_attack(siamese_net, predictor_net, ae_model, args)

    # train_ae_model(siamese_net, predictor_net, ae_model, args, 100)
    # test_siamese(siamese_net, predictor_net, ae_model, args)

    # save_siamese(siamese_net, predictor_net, args)

    # eval_iters(ae_model, dis_model)

    print("Done!")
