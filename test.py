from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch
import json
import pprint
import sys

from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
from tvr.models.modeling import DRL, AllGather
from tvr.models.optimization import BertAdam
from tvr.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

allgather = AllGather.apply

global logger


def get_args(description='Disentangled Representation Learning for Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--wti_arch', type=int, default=2, help="select a architecture for weight branch")

    parser.add_argument('--cdcr', type=int, default=3, help="channel decorrelation regularization")
    parser.add_argument('--cdcr_alpha1', type=float, default=1.0, help="coefficient 1")
    parser.add_argument('--cdcr_alpha2', type=float, default=0.06, help="coefficient 2")
    parser.add_argument('--cdcr_lambda', type=float, default=0.001, help="coefficient for cdcr")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank) # args set it as cpu, but will be changed here :D
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = DRL(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length, test_video_dict, test_sentence_dict = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length, val_video_dict, val_sentence_dict = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length, val_video_dict, val_sentence_dict = test_dataloader, test_length, test_video_dict, test_sentence_dict

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length, test_video_dict, test_sentence_dict = val_dataloader, val_length, val_video_dict, val_sentence_dict 

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dt %dv", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dt %dv", val_length[0], val_length[1])

    return test_dataloader, test_video_dict, test_sentence_dict 


# model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v
def _run_on_single_gpu(model, t_mask_list, v_mask_list, t_feat_list, v_feat_list, mini_batch=32):

    sim_matrix = []
    logger.info('[start] map to main gpu')

    batch_t_mask = torch.split(t_mask_list, mini_batch)
    batch_v_mask = torch.split(v_mask_list, mini_batch)
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)

    logger.info('[finish] map to main gpu')
    with torch.no_grad():
        for idx1, (t_mask, t_feat) in enumerate(zip(batch_t_mask, batch_t_feat)):
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(batch_v_mask, batch_v_feat)):
                b1b2_logits, *_tmp = model.get_similarity_logits(t_feat, v_feat, t_mask, v_mask)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    return sim_matrix


def preprocess_text(text_input, max_words=32):
    tokenizer = ClipTokenizer()
    words = tokenizer.tokenize(text_input)
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                     "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    input_ids = tokenizer.convert_tokens_to_ids(words)
    t_tokens = [tokenizer.decode([_t_id]) for _t_id in input_ids]
    t_tokens = [t.replace('<|startoftext|>', '[CLS]').replace('<|endoftext|>', '[SEP]').strip() for t in t_tokens]

    return t_tokens, torch.tensor(input_ids)

def eval_epoch(args, model, test_dataloader, device, text_mask, text_feat):
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, ids_v = [], [], [], [], []

    with torch.no_grad():
        
        logger.info('[start] extract text+video feature')
        # only video features
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            _, _, video, video_mask, inds = batch
            video_feat = model.get_video_feat(video, video_mask)
            ids_v.append(inds)
            batch_mask_v.append(video_mask)
            batch_feat_v.append(video_feat)

        batch_mask_t.append(text_mask)
        batch_feat_t.append(text_feat)

        ids_v = allgather(torch.cat(ids_v, dim=0), args).squeeze()

        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
        batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
        batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
        batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)

        batch_mask_t = batch_mask_t.unsqueeze(0)

        batch_mask_v[ids_v] = batch_mask_v.clone()
        batch_feat_v[ids_v] = batch_feat_v.clone()

        batch_mask_t = batch_mask_t.clone()
        batch_feat_t = batch_feat_t.clone()

        batch_mask_v = batch_mask_v[:ids_v.max() + 1, ...]
        batch_feat_v = batch_feat_v[:ids_v.max() + 1, ...]

        logger.info('[finish] extract text+video feature')


    logger.info('{} {} {} {}'.format(len(batch_mask_t), len(batch_mask_v), len(batch_feat_t), len(batch_feat_v)))

    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        sim_matrix = _run_on_single_gpu(model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    logger.info('[end] calculate the similarity')
    
    return sim_matrix, ids_v, batch_mask_v, batch_feat_v

# not use (나중에 feat 저장하게 되는 경우 사용)
def eval_epoch_with_saved_batch_v(args, model, test_dataloader, device, text_mask, text_feat, batch_mask_v, batch_feat_v, ids_v):
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    batch_mask_t, batch_feat_t = [], []

    # ----------------------------
    # 1. cache the features
    # ----------------------------
    with torch.no_grad():

        logger.info('[start] extract text+video feature')

        batch_mask_t.append(text_mask)
        batch_feat_t.append(text_feat)

        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
        batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)

        batch_mask_t = batch_mask_t.unsqueeze(0)

        batch_mask_t = batch_mask_t.clone()
        batch_feat_t = batch_feat_t.clone()

        logger.info('[finish] extract text+video feature')

    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        sim_matrix = _run_on_single_gpu(model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    logger.info('[end] calculate the similarity')

    return sim_matrix, ids_v

def return_sorted_rank_sim_matrix(sim_m, video_names, video_captions):
    sim_dict = dict()
    for video_name, similarity in zip(video_names, video_captions, sim_m[0]):
        sim_dict[video_name] = float(similarity)
    
    sorted_sim_dict = sorted(sim_dict.items(), key = lambda x : x[1], reverse = True) # list

    sorted_rank_sim_dict = dict()
    for i, data in enumerate(sorted_sim_dict, 1):
        sorted_rank_sim_dict[i] = {'video_id' : data[0], 'similarity' : data[1]}
    return sorted_rank_sim_dict

def return_text_mask_feat(model, args, text_input):
    max_words= args.max_words
    _, text = preprocess_text(text_input, max_words)
    text_mask = (text > -1).type(torch.long)

    use_padding = False
    if use_padding:
        text_pad = torch.tensor(np.zeros(max_words - len(text))).type(torch.long)
        text_mask = torch.cat([text_mask, text_pad])
        text = torch.cat([text, text_pad])
        assert len(text_mask) == max_words
        assert len(text) == max_words

    text = text.to(args.device)
    text_mask = text_mask.to(args.device)
    text_feat = model.get_text_feat(text, text_mask)
    return text_mask, text_feat

def save_output(sorted_rank_sim_dict, file_name, args):
    save_file = True
    if save_file :
        output_json_path = args.output_dir + f'/{file_name}.json'
        with open(output_json_path, 'w') as f:
            json.dump(sorted_rank_sim_dict, f)

def print_topN_of_matrix(sorted_rank_sim_dict, top_n):
    print('Top ', top_n)
    for i in range(top_n):
        print(sorted_rank_sim_dict[i])

def main():

    #### default ####################################################
    global logger
    args = get_args()
    args.output_dir = args.output_dir
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)
    args = set_seed_logger(args)
    model = build_model(args)
    # breakpoint()
    test_dataloader, test_video_dict, test_sentence_dict = build_dataloader(args)

    save_output(test_sentence_dict, 'test_sentence_dict', args)

    #### default ####################################################

    #### initial test ###############################################

    #### get text input ########################################
    print('Enter a type (t for text, i for idx) : i')
    print('Enter a index number from 0 to 999: 0')
    input_type = 'i'
    idx = 0
    text_input = test_sentence_dict[idx][1][0]

    # input_type = input('Enter a type (t for text, i for idx) : ')
    # if input_type == 't':
    #     print('Enter a sentence:')
    #     text_input = sys.stdin.readline().strip()
    # elif input_type =='i':
    #     print('Enter a index number from 0 to 999: ')
    #     idx = int(input())
    #     text_input = test_sentence_dict[idx][1][0]
    # else:
    #     breakpoint()
    ############################################################
    
    text_mask, text_feat = return_text_mask_feat(model, args, text_input)

    # similarity matrix 생성
    sim_m, ids_v, batch_mask_v, batch_feat_v = eval_epoch(args, model, test_dataloader, args.device, text_mask, text_feat)

    video_dict_list = list()
    for id_v, sim in zip(ids_v, sim_m[0]):
        new_dict = {"video_id" : test_sentence_dict[id_v.item()][0], "video_caption" : test_sentence_dict[id_v.item()][1][0], "similarity" : sim, "idx" : id_v.item()}
        video_dict_list.append(new_dict)

    sorted_rank_dict = sorted(video_dict_list, key = lambda x : x['similarity'], reverse = True)
    sorted_idx_dict = sorted(video_dict_list, key = lambda x : x['idx'], reverse = False)

    # Top_N 개 출력
    top_n = 5
    print('Text: ', text_input)
    print_topN_of_matrix(sorted_rank_dict, top_n)

    ####### print rank ############################################
    if input_type == 'i':
        test_idx_rank = sorted_rank_dict.index(sorted_idx_dict[idx])
        print('test input: ', text_input)
        print('rank: ', test_idx_rank)
        print('info: ', sorted_idx_dict[idx])
    ####### print rank ############################################

    ################ initial test ##########################

    ################ more test #############################
    n_times = 100 # test 용으로 100번 도는 코드
    for _ in range(n_times):
        # print('Enter a sentence:')
        # text_input = sys.stdin.readline().strip()

        #### get text input ########################################
        input_type = input('Enter a type (t for text, i for idx) :')
        if input_type == 't':
            print('Enter a sentence: ')
            text_input = sys.stdin.readline().strip()
        elif input_type =='i':
            print('Enter a index number from 0 to 999: ')
            idx = int(input())
            text_input = test_sentence_dict[idx][1][0]
        else:
            breakpoint()
        ############################################################
            

        text_mask, text_feat = return_text_mask_feat(model, args, text_input)
        sim_m, ids_v = eval_epoch_with_saved_batch_v(args, model, test_dataloader, args.device, text_mask, text_feat, batch_mask_v, batch_feat_v, ids_v)
        video_dict_list = list()
        for id_v, sim in zip(ids_v, sim_m[0]):
            new_dict = {"video_id" : test_sentence_dict[id_v.item()][0], "video_caption" : test_sentence_dict[id_v.item()][1][0], "similarity" : sim, "idx" : id_v.item()}
            video_dict_list.append(new_dict)
        sorted_rank_dict = sorted(video_dict_list, key = lambda x : x['similarity'], reverse = True)
        sorted_idx_dict = sorted(video_dict_list, key = lambda x : x['idx'], reverse = False)

        top_n = 5
        print('Text: ', text_input)
        print_topN_of_matrix(sorted_rank_dict, top_n)

        ####### print rank ############################################
        if input_type == 'i':
            test_idx_rank = sorted_rank_dict.index(sorted_idx_dict[idx])
            print('test input: ', text_input)
            print('rank:', test_idx_rank)
            print('info:', sorted_idx_dict[idx])
        ####### print rank ############################################

    ################ more test #############################

    print('end')

if __name__ == "__main__":
    main()