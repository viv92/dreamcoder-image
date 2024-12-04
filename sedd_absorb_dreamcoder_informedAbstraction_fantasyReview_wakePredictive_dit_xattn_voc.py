'''
### Program implementing 

## Features:
1. 

## Todos / Questions:
1.

'''

import os
import cv2
import math 
from copy import deepcopy 
from matplotlib import pyplot as plt 
import numpy as np
import torch
torch.set_float32_matmul_precision('high') # use TF32 precision for speeding up matmul
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

# import T5 
from transformers import T5Tokenizer, T5ForConditionalGeneration

# import CLIP and Caption models
from transformers import CLIPProcessor, CLIPModel, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# import DiT 
from utils_sedd_dreamcoder_informedAbstraction_fantasyReview_wakePredictive_dit_xattn import *


# utility function to load img and captions data 
def load_data(size):
    imgs_folder = '/home/vivswan/experiments/muse/dataset_coco_val2017/images/'
    captions_file_path = '/home/vivswan/experiments/muse/dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_pairs = {}, []
    print('Loading Data...')
    num_iters = len(captions['images']) + len(captions['annotations'])
    pbar = tqdm(total=num_iters)

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # use img_name as key for img_cap_dict
        img_filename = img_dict[id]

        # load image from img path 
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 0, 1) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            # NOTE: no random resizing as its asking the model to learn to predict different img tokens for the same caption
            # torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
        ])
        img = transforms(img)

        # img_cap_pairs.append([img_filename, caption])
        img_cap_pairs.append([img, caption])
        if size > 0 and len(img_cap_pairs) > size:
            break
        pbar.update(1)
    pbar.close()
    return img_cap_pairs


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - then obtain embeddings for them
def process_batch(minibatch, tokenizer, img_size, device):
    augmented_imgs, captions = list(map(list, zip(*minibatch)))

    # augmented_imgs = []
    # img_files, captions = list(map(list, zip(*minibatch)))

    # tokenize captions 
    caption_tokens_dict = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)

    # # get augmented imgs
    # imgs_folder = '/home/vivswan/experiments/muse/dataset_coco_val2017/images/'
    # for img_filename in img_files:
    #     img_path = imgs_folder + img_filename
    #     img = cv2.imread(img_path, 1)
    #     resize_shape = (img_size, img_size)
    #     img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    #     img = np.float32(img) / 255
    #     img = torch.tensor(img)
    #     img = img.permute(2, 0, 1) # [w,h,c] -> [c,h,w]
    #     transforms = torchvision.transforms.Compose([
    #         # NOTE: no random resizing as its asking the model to learn to predict different img tokens for the same caption
    #         # torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
    #         # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
    #         # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
    #     ])
    #     img = transforms(img)
    #     augmented_imgs.append(img)

    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    caption_tokens_dict = caption_tokens_dict.to(device)
    return augmented_imgs, caption_tokens_dict


# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval() 

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer
        
# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)


def reconstruction_loss(recon_x, x):
    criterion = nn.MSELoss(reduction='mean')
    # reconstruction loss
    recon_loss = criterion(recon_x, x)
    return recon_loss

def compression_loss(z):
    criterion = nn.MSELoss(reduction='mean')
    # latent compress loss - drive latents to zero (pad) latent
    compress_targets = torch.zeros_like(z)
    compress_loss = criterion(z, compress_targets)
    return compress_loss
        

# utility function to convert img to sequence of patches (used to process img data)
def img_to_patch_seq(img, patch_size, num_latents):
    b,c,h,w = img.shape
    p = patch_size 
    assert (h % p == 0) and (w % p == 0)
    seq_len = (h * w) // (p * p)
    assert num_latents == seq_len 
    n_rows, n_cols = h//p, w//p
    patch_seq = []
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * p
            j = col * p
            patch = img[:, :, i:i+p, j:j+p]
            patch = torch.flatten(patch, start_dim=1, end_dim=-1) # patch.shape: [b, patch_dim]
            patch_seq.append(patch)
    patch_seq = torch.stack(patch_seq, dim=0) # patch_seq.shape: [seq_len, b, patch_dim]
    patch_seq = patch_seq.transpose(0, 1) # patch_seq.shape: [b, seq_len, patch_dim]
    # assert seq_len == patch_seq.shape[1]
    # assert patch_dim == patch_seq.shape[-1]
    return patch_seq

# utility function to convert sequence of patches to an img (used to process img data)
def patch_seq_to_img(x, patch_size, channels): # x.shape: [batch_size, max_seq_len, patch_dim]
    batch_size, seq_len, patch_dim = x.shape[0], x.shape[1], x.shape[2]
    p = patch_size 
    x = x.permute(1, 0, 2) # x.shape: [seq_len, batch_size, patch_dim]
    x = x.reshape(seq_len, batch_size, channels, p*p)
    x = x.reshape(seq_len, batch_size, channels, p, p)
    nrows = int(math.sqrt(seq_len))
    ncols = nrows
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                row = x[i * nrows + j]
            else:
                row = torch.cat((row, x[i * nrows + j]), dim=-1)
        if i == 0:
            imgs = row # row.shape: [batch_size, 3, p, w]
        else:
            imgs = torch.cat((imgs, row), dim=-2) # imgs.shape: [batch_size, 3, h, w]
    # imgs = imgs.permute(0, 2, 3, 1) # imgs.shape: [batch_size, h, w, 3]
    return imgs 


# convert tensor to img
def to_img(x):
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.clamp(0, 1) # clamp img to be strictly in [-1, 1]
    x = x.permute(0,2,3,1) # [b,c,h,w] -> [b,h,w,c]
    return x 

# function to save a generated img
def save_img_generated(x_g, save_path):
    gen_img = x_g.detach().cpu().numpy()
    gen_img = np.uint8( gen_img * 255 )
    # bgr to rgb 
    # gen_img = gen_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)

# function to save a test img and its reconstructed img 
def save_img_reconstructed(x, x_r, save_path):
    concat_img = torch.cat([x, x_r], dim=1)
    concat_img = concat_img.detach().cpu().numpy()
    concat_img = np.uint8( concat_img * 255 )
    # bgr to rgb 
    # concat_img = concat_img[:, :, ::-1]
    cv2.imwrite(save_path, concat_img)


def ema(arr, val, r=0.01):
    if len(arr) == 0:
        return [val]
    newval = arr[-1] * (1-r) + val * r 
    arr.append(newval)
    return arr 


### main
if __name__ == '__main__':

    # hyperparams for quantization
    num_quantized_values = [7, 5, 5, 5, 5] # L in fsq paper
    latent_dim = len(num_quantized_values)
    img_size = 64 # 128 # voc 
    img_channels = 3 
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    img_latent_dim = latent_dim # as used in the pretrained VQVAE 

    patch_size = 16 # necessary that img_size % patch_size == 0
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    # hyperparams for FSQ Transformer
    d_model_fsq = 256 # patch_dim * 1  
    n_heads_fsq = 8
    assert d_model_fsq % n_heads_fsq == 0
    d_k_fsq = d_model_fsq // n_heads_fsq 
    d_v_fsq = d_k_fsq 
    n_layers_fsq = 6
    d_ff_fsq = d_model_fsq * 4

    # hyperparams for Predictive Scorer
    d_model_ps = 256
    n_heads_ps = 8
    assert d_model_ps % n_heads_ps == 0
    d_k_ps = d_model_ps // n_heads_ps 
    d_v_ps = d_k_ps 
    n_layers_ps = 6
    d_ff_ps = d_model_ps * 4

    # hyperparams for T5 (T5 decoder implements the consistency model backbone)
    d_model_t5 = 768 # d_model for T5 (required for image latents projection)
    max_seq_len_t5 = 512 # required to init T5 Tokenizer
    # dropout = 0. # TODO: check if we can set the dropout in T5 decoder

    # hyperparams for dit
    d_model = 512
    n_layers = 6
    n_heads = 4 # might be better to keep this low when modeling image token sequences (since small seqlen)
    d_k = d_model // n_heads
    d_v = d_k 
    d_ff = d_model * 4

    dropout = 0.1
    weight_decay = 0.1 
    compress_factor = 0.01 # 0.005 
    reconstruction_factor = 1
    prediction_factor = 0.1 

    clip_score_threshold_wake = 30.0 # 33.0
    clip_score_threshold_fantasy = 20.0 

    data_size = 1000
    test_fraction = 0.75 # majority is test data in the beginning

    # get vocab size and mask token for sedd
    vocab_size = 1
    for n in num_quantized_values:
        vocab_size *= n 
    vocab_size += 1 # for mask token
    mask_token = vocab_size - 1

    # hyperparams for sleep mode
    sleep_mode = 1 # 0 = wake, 1 = sleep, 2 = dream      
    sleep_steps = 100
    wake_steps = 16
    dream_steps = sleep_steps
    num_switches = -1
    sleep_steps_list = [wake_steps, sleep_steps, dream_steps]

    # hyperparams for training 
    diffusion_start_time_eps = 1e-3
    batch_size = 512
    gradient_accumulation_steps = 1 # TODO: why does the loss curve become flat (instead of going down) on increasing this ?
    lr = 3e-4 # 1e-4
    num_cycles = 2000
    num_train_steps = sum(sleep_steps_list) * num_cycles
    train_steps_done = 0
    random_seed = 10
    resume_training_from_ckpt = False   

    # hyperparams for sampling
    num_sampling_steps = seq_len #* 2
    dream_visualize_sample_batch_size = 4
    wake_batch_size = 128 # 32 
    p_uncond = 0.1 
    cfg_scale = 2.0

    # hyperparams for figures and plotting
    sampling_freq = 360 # 720
    plot_freq = sampling_freq * 4
    first_plot = [True, True, True]

    # hyperparams for caption (fantasy) model
    max_caption_length = 16
    num_caption_beams = 4
    caption_gen_kwargs = {"max_length": max_caption_length, "num_beams": num_caption_beams}
    max_fantasies = 512
    fantasy_batch_size = 128 # 32
    fantasy_tries = int(max_fantasies / fantasy_batch_size)
    delay_start_iter = 6 * 1000
    ps_model_train_tries = int( (sleep_steps / wake_steps) * (batch_size / wake_batch_size) )
    fantasy_gen_cycle_counter = -1

    # create hyperparam str
    hyperparam_dict = {}
    # hyperparam_dict['method'] = 'seddA_dreamcoder_infAbs_Fant_shuffled_voc' 
    # hyperparam_dict['method'] = 'seddA_dreamcoder_FantReview_WakePredFix4_voc' 
    hyperparam_dict['method'] = 'seddA_dreamcoder_FantReview4_noWakePred_incClipT_voc' 
    # hyperparam_dict['seqlen'] = seq_len
    # hyperparam_dict['D_patch'] = patch_dim
    hyperparam_dict['D_fsq'] = d_model_fsq
    hyperparam_dict['D_dit'] = d_model  
    hyperparam_dict['B'] = batch_size 
    hyperparam_dict['lr'] = lr
    hyperparam_dict['Wdecay'] = weight_decay
    hyperparam_dict['dropout'] = dropout
    hyperparam_dict['CF'] = compress_factor
    hyperparam_dict['RF'] = reconstruction_factor
    hyperparam_dict['PF'] = prediction_factor
    hyperparam_dict['clipTW'] = clip_score_threshold_wake 
    hyperparam_dict['clipTF'] = clip_score_threshold_fantasy 
    # hyperparam_dict['initMode'] = sleep_mode
    # hyperparam_dict['sleep'] = sleep_steps
    # hyperparam_dict['dream'] = dream_steps
    hyperparam_dict['wake'] = wake_steps
    # hyperparam_dict['sampSteps'] = num_sampling_steps
    hyperparam_dict['testFrac'] = test_fraction
    # hyperparam_dict['dataSz'] = data_size 
    hyperparam_dict['delay'] = delay_start_iter 
    hyperparam_dict['maxF'] = max_fantasies 
    # hyperparam_dict['Ptries'] = ps_model_train_tries 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + '=' + str(v)


    results_dir = './results' + hyperparam_str + '/'
    ckpts_dir = './ckpts/'
    dit_ckpt_path = ckpts_dir + 'dit_' + hyperparam_str + '.pt'
    fsq_ckpt_path = ckpts_dir + 'fsq_' + hyperparam_str + '.pt'
      
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)


    # t5 model (for encoding captions) 
    t5_model_name = 't5-base'

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create dataset from img_cap_dict
    dataset = load_data(data_size)

    # shuffle and split dataset into train and test
    # idx = np.arange(len(dataset))
    # np.random.shuffle(idx)
    # dataset = [dataset[i] for i in idx]
    split_index = int(test_fraction * len(dataset)) 
    test_dataset = dataset[:split_index] # majority data is in test set
    dataset = dataset[split_index:]

    # load FSQ 
    # init transformer encoder
    encoder_transformer = init_fsq_encoder_transformer(patch_dim, d_model_t5, seq_len, d_model_fsq, d_k_fsq, d_v_fsq, n_heads_fsq, n_layers_fsq, d_ff_fsq, dropout, latent_dim, device)
    # init transformer decoder
    decoder_transformer = init_fsq_decoder_transformer(latent_dim, seq_len, d_model_fsq, d_k_fsq, d_v_fsq, n_heads_fsq, n_layers_fsq, d_ff_fsq, dropout, patch_dim, device)
    # init FSQ_Transformer 
    fsq = FSQ_Transformer(device, num_quantized_values, encoder_transformer, decoder_transformer, seq_len).to(device)

    # init optimizer
    fsq_optimizer = torch.optim.AdamW(fsq.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    
    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len_t5)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)
    # delete t5_decoder to save ram 
    del t5_model.decoder 

    # init pretrained clip model - verifier for wake mode
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)

    # init caption model - to generate fantasies 
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_model = caption_model.to(device)

    # init dit
    x_seq_len = seq_len 
    # condition_seq_len = 1 # just the class label
    max_seq_len = seq_len + 1 # [t, x]
    # x_dim = 1
    condition_dim = d_model_t5
    dit = init_dit(max_seq_len, x_seq_len, d_model, condition_dim, vocab_size, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # freeze t5_encoder
    freeze(t5_model.encoder)

    # optimizer and loss criterion
    dit_optimizer = torch.optim.AdamW(params=dit.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # init predictive scorer 
    ps_model = init_predictive_scorer(patch_dim, d_model_t5, seq_len, d_model_ps, d_k_ps, d_v_ps, n_heads_ps, n_layers_ps, d_ff_ps, dropout, 1, device).to(device)
    ps_optimizer = torch.optim.AdamW(ps_model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # load ckpt
    if resume_training_from_ckpt:
        fsq, fsq_optimizer = load_ckpt(fsq_ckpt_path, dit, fsq_optimizer, device=device, mode='train')
        dit, dit_optimizer = load_ckpt(dit_ckpt_path, dit, dit_optimizer, device=device, mode='train')

    # for plotting 
    results_dit_loss = []
    results_train_loss, results_recon_loss, results_compress_loss = [], [], []
    results_codebook_usage, results_codebook_unique, results_masked_latents = [], [], []
    results_test_items, results_num_fantasies = [len(test_dataset)], [0]
    results_sleep_ps_loss, results_wake_ps_loss, results_sleep_fantasy_loss = [], [], []

    # train

    train_step = train_steps_done
    sleep_mode_counter = 0 
    pbar = tqdm(total=num_train_steps)
    wake_visualize_dataset = [] # for visualizing imgs generated during wake
    wake_dataset = [] # used for prediction loss 
    solved_test_idx = [] # used to remove solved test items from test dataset
    fantasy_dataset = [] 
    
    while train_step < num_train_steps + train_steps_done:

        # handle sleep mode 
        if not (num_switches == 0):
            switch_required = (sleep_mode_counter == sleep_steps_list[sleep_mode])

            # if switch from wake to sleep, remove test items solved in wake phase
            if switch_required and (sleep_mode == 0):
                solved_test_idx = list(set(solved_test_idx)) # deduplicate
                test_dataset = [x for i,x in enumerate(test_dataset) if i not in solved_test_idx]
                solved_test_idx = []
                remaining_test_items = len(test_dataset)
                results_test_items.append(remaining_test_items)

            # if switch from sleep to dream, flush wake dataset
            if switch_required and (sleep_mode == 1):
                wake_dataset = []

            while switch_required:
                sleep_mode += 1
                sleep_mode = sleep_mode % len(sleep_steps_list)
                sleep_mode_counter = 0
                switch_required = (sleep_mode_counter == sleep_steps_list[sleep_mode]) 
                num_switches -= 1


        if (sleep_mode == 0) and (len(test_dataset) > 0): # wake mode - get solved test data 
            dit.eval()
            fsq.eval()

            # fetch test minibatch
            test_idx = np.arange(len(test_dataset))
            np.random.shuffle(test_idx)
            test_idx = test_idx[:wake_batch_size]
            minibatch = [test_dataset[i] for i in test_idx]

            _, captions = list(map(list, zip(*minibatch))) # raw captions required for clip

            # process minibatch 
            test_imgs, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device)

            with torch.no_grad():

                # extract cap tokens and attn_mask from cap_tokens_dict
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                # feed cap_tokens to t5 encoder to get encoder output
                t5_enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # t5_enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

                # get sample tokens corresponding to indices of codebook 
                x_sample = get_sample(dit, seq_len, mask_token, vocab_size, num_sampling_steps, t5_enc_out.shape[0], t5_enc_out, cfg_scale, device) # x_sample.shape: [b, seqlen]
                x_sample = x_sample.flatten() # shape: [b * seqlen]

                # get codebook vectors indexed by x_sample
                sampled_img_latents = fsq.codebook[x_sample] # sampled_img_latents.shape: [b, seqlen, latent_dim]
                gen_img_patch_seq, _ = fsq.decode(sampled_img_latents.float())
                # convert patch sequence to img 
                gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

                # clip requires imgs to be in [0,255]
                imgs_clip = to_img(gen_imgs)
                imgs_clip = (imgs_clip * 255).int()
                imgs_clip = imgs_clip[:, :, :, [2,1,0]] # bgr -> rgb

                # pre-process inputs for clip
                inputs = clip_processor(text=captions, images=imgs_clip, return_tensors="pt", padding=True)
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)

                # get clip scores and threshold
                outputs = clip_model(**inputs)
                logits_per_text = outputs.logits_per_text  # this is the image-text similarity score
                clip_scores = torch.diag(logits_per_text)
                preds = torch.gt(clip_scores, clip_score_threshold_wake).int()
                idx = torch.where(preds)[0]

            # add solved test items to train dataset (with new generated imgs) - these are permanently solved
            for i in idx:
                gen_img = gen_imgs[i].cpu()
                cap = captions[i]
                dataset.append([gen_img, cap])

                # for visualization
                test_img = test_imgs[i].cpu()
                wake_visualize_dataset.append([test_img, gen_img, cap])
            
            # update solved_test_idx
            solved_test_idx.extend([test_idx[i] for i in idx])

            # # add to wake dataset 
            # for i in range(len(captions)):
            #     gen_img = gen_imgs[i].cpu()
            #     cap = captions[i]
            #     wake_dataset.append([gen_img, cap])

            # # train step for predictive scorer 
            # for i in range(ps_model_train_tries):
            #     ps_scores = ps_model(gen_img_patch_seq, t5_enc_out) # [batch_size, 1]
            #     ps_loss = F.mse_loss(ps_scores.squeeze(-1), clip_scores, reduction='mean')
            #     ps_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(ps_model.parameters(), max_norm=1.0)
            #     ps_optimizer.step()
            #     ps_optimizer.zero_grad()

            # results_wake_ps_loss.append(ps_loss.item())

            pbar.update(1)
            # pbar.set_description('mode:{} test_items:{} ps_loss:{:.3f}'.format(sleep_mode, len(test_dataset), ps_loss.item()))
            pbar.set_description('mode:{} test_items:{}'.format(sleep_mode, len(test_dataset)))
            dit.train()
            fsq.train()
    

        if sleep_mode == 1: # sleep mode - train FSQ 
            dit.eval()

            # alternatively train on train_dataset and wake_dataset
            sleep_dataset = dataset 
            sleep_on_what = 0
            # if (train_step > delay_start_iter) and (len(fantasy_dataset) > 0) and (len(wake_dataset) > 0):
            if (train_step > delay_start_iter) and (len(fantasy_dataset) > 0):
                sleep_on_what = train_step % 2 # 0 = train_dataset, 1 = fantasy_dataset, 2 = wake_dataset
                if sleep_on_what == 1:
                    sleep_dataset = fantasy_dataset 
                if sleep_on_what == 2:
                    sleep_dataset = wake_dataset

            # fetch minibatch
            idx = np.arange(len(sleep_dataset))
            np.random.shuffle(idx)
            idx = idx[:batch_size]
            minibatch = [sleep_dataset[i] for i in idx]

            _, captions = list(map(list, zip(*minibatch))) # raw captions required for clip (prediction loss)

            # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
            # note that we don't create embeddings yet, since that's done by the image_encoder and T5 model
            imgs, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device) # imgs.shape:[batch_size, 3, 32, 32], captions.shape:[batch_size, max_seq_len]

            with torch.no_grad():

                # extract cap tokens and attn_mask from cap_tokens_dict
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                # feed cap_tokens to t5 encoder to get encoder output
                t5_enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # t5_enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

                # convert img to sequence of patches
                x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]

            # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            # forward prop through FSQ 
            recon_x, z_e, z_q, usage, unique, percent_masked_latents = fsq(x, t5_enc_out) # recon_x.shape: [b, seq_len, patch_dim]

            # calculate loss

            prediction_loss = 0
            # if (train_step > delay_start_iter):
            #     ps_scores = ps_model(recon_x, t5_enc_out) # [batch_size, 1]
            #     ps_scores = ps_scores / 100 # normalize ps_scores to range [0, 1]
            #     prediction_loss = prediction_factor * -1 *  ps_scores.mean() 

            if sleep_on_what == 2: # wake_dataset
                loss = prediction_loss
            else:
                recon_loss = reconstruction_factor * reconstruction_loss(recon_x, x)
                compress_loss = compress_factor * compression_loss(z_e)
                loss = recon_loss + compress_loss + prediction_loss

            loss.backward()
            # gradient cliping 
            torch.nn.utils.clip_grad_norm_(fsq.parameters(), max_norm=1.0)
            # gradient step
            fsq_optimizer.step()
            fsq_optimizer.zero_grad()

            if sleep_on_what == 2:
                results_sleep_ps_loss.append(prediction_loss.item())
            elif sleep_on_what == 1: 
                results_sleep_fantasy_loss.append(loss.item())
            else:
                results_codebook_usage.append(usage.item())
                results_codebook_unique.append(unique)
                results_masked_latents.append(percent_masked_latents.item())
                results_train_loss.append(loss.item())
                results_recon_loss.append(recon_loss.item())
                results_compress_loss.append(compress_loss.item())
                if (train_step < delay_start_iter):
                    results_sleep_fantasy_loss.append(0)
                    results_sleep_ps_loss.append(0)


            pbar.update(1)
            pbar.set_description('mode:{} loss: {:.3f}'.format(sleep_mode, loss.item()))
            dit.train()


        if sleep_mode == 2: # dream mode - train DiT
            fsq.eval()

            ## generate fantasy dataset 

            if (sleep_mode_counter == 0) and (train_step > delay_start_iter):
                fantasy_gen_cycle_counter += 1
                if fantasy_gen_cycle_counter % 4 == 0:
                    with torch.no_grad():

                        fantasy_dataset = [] # flush old data
                        fantasy_scores = []
                        # pbar2 = tqdm(total = fantasy_tries)

                        for j in range(fantasy_tries):

                            if len(fantasy_dataset) >= max_fantasies:
                                break

                            # fetch test minibatch for test captions
                            test_idx = np.arange(len(test_dataset))
                            np.random.shuffle(test_idx)
                            test_idx = test_idx[:fantasy_batch_size]
                            minibatch = [test_dataset[i] for i in test_idx]

                            _, captions = list(map(list, zip(*minibatch))) # raw captions required for clip

                            # process minibatch 
                            _, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device)

                            # extract cap tokens and attn_mask from cap_tokens_dict
                            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                            # feed cap_tokens to t5 encoder to get encoder output
                            t5_enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # t5_enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

                            # get sample tokens corresponding to indices of codebook 
                            x_sample = get_sample(dit, seq_len, mask_token, vocab_size, num_sampling_steps, t5_enc_out.shape[0], t5_enc_out, cfg_scale, device) # x_sample.shape: [b, seqlen]
                            x_sample = x_sample.flatten() # shape: [b * seqlen]

                            # get codebook vectors indexed by x_sample
                            sampled_img_latents = fsq.codebook[x_sample] # sampled_img_latents.shape: [b, seqlen, latent_dim]
                            gen_img_patch_seq, _ = fsq.decode(sampled_img_latents.float())
                            # convert patch sequence to img 
                            gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

                            # clip requires imgs to be in [0,255]
                            imgs_clip = to_img(gen_imgs)
                            imgs_clip = (imgs_clip * 255).int()
                            imgs_clip = imgs_clip[:, :, :, [2,1,0]] # bgr -> rgb

                            # generate captions
                            pixel_values = caption_feature_extractor(images=imgs_clip, return_tensors="pt").pixel_values
                            pixel_values = pixel_values.to(device)
                            output_ids = caption_model.generate(pixel_values, **caption_gen_kwargs)
                            gen_captions = caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                            gen_captions = [caption_pred.strip() for caption_pred in gen_captions]

                            # evaluate generated captions
                            inputs = clip_processor(text=gen_captions, images=imgs_clip, return_tensors="pt", padding=True)
                            for k in inputs.keys():
                                inputs[k] = inputs[k].to(device)
                            outputs = clip_model(**inputs)
                            logits_per_text = outputs.logits_per_text  # this is the image-text similarity score
                            scores = torch.diag(logits_per_text)
                            preds = torch.gt(scores, clip_score_threshold_fantasy).int()
                            idx = torch.where(preds)[0]

                            # add to fantasy dataset
                            for i in idx:
                                gen_img = gen_imgs[i].cpu()
                                gen_cap = gen_captions[i]
                                fantasy_dataset.append([gen_img, gen_cap])
                                fantasy_scores.append(scores[i].item())

                            # pbar2.update(1)
                            # pbar2.set_description('mode:{} num_fantasies: {}'.format(sleep_mode, len(fantasy_dataset)))

                        results_num_fantasies.append(len(fantasy_dataset))

                        # # update clip thresholds to top quartile score
                        # if len(fantasy_scores) > 0:
                        #     fantasy_scores.sort()
                        #     quartile = fantasy_scores[-int(len(fantasy_scores)/4):]
                        #     quartile_score = int( sum(quartile) / len(quartile) )
                        #     clip_score_threshold_fantasy = quartile_score
                        #     if clip_score_threshold_fantasy > clip_score_threshold_wake:
                        #         clip_score_threshold_wake = clip_score_threshold_fantasy

                        # update clip thresholds 
                        if len(fantasy_dataset) > int(max_fantasies * 0.75):
                            clip_score_threshold_fantasy += 1
                            if clip_score_threshold_fantasy > clip_score_threshold_wake:
                                clip_score_threshold_wake = clip_score_threshold_fantasy


            ## back to dream mode training

            # alternatively train on replays and fantasies according to data proportions
            dream_dataset = dataset 
            if (len(fantasy_dataset) > 0):
                # r = int(fantasy_ratio_numerator / len(fantasy_dataset))
                # if train_step % (r+1) == 0:
                if train_step % 2 == 0:
                    dream_dataset = fantasy_dataset

            # fetch minibatch
            idx = np.arange(len(dream_dataset))
            np.random.shuffle(idx)
            idx = idx[:batch_size]
            minibatch = [dream_dataset[i] for i in idx]

            # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
            # note that we don't create embeddings yet, since that's done by the image_encoder and T5 model
            imgs, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device) # imgs.shape:[batch_size, 3, 32, 32], captions.shape:[batch_size, max_seq_len]

            with torch.no_grad():

                # extract cap tokens and attn_mask from cap_tokens_dict
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                # feed cap_tokens to t5 encoder to get encoder output
                t5_enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # t5_enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

                # convert img to sequence of patches
                x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]
                # obtain img latent embeddings using pre-trained VQVAE
                z_e = fsq.encode(x, t5_enc_out) # z_e.shape: [b, seq_len,  img_latent_dim]
                img_latents, _, _, _, _, target_idx = fsq.quantize(z_e) # target_idx.shape: [b * img_latent_seqlen]
                target_idx = target_idx.view(-1, seq_len) # [b, seqlen] 


            x = target_idx # x.shape: [b, seq_len] 
            condition = t5_enc_out

            # for sampling 
            sample_caption_emb = condition[:1] # NOTE that we sample one class label but generate n_sample imgs for that label
            sample_caption_emb = sample_caption_emb.expand(dream_visualize_sample_batch_size, -1, -1)

            # set condition = None with prob p_uncond
            if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
                condition = None

            # sample diffusion time ~ uniform(eps, 1)
            t = (1 - diffusion_start_time_eps) * torch.rand(x.shape[0], device=device) + diffusion_start_time_eps

            # get noise from noise schedule
            sigma, dsigma = logLinearNoise(t)

            # perturb the data
            x_perturb = perturb(x, sigma, mask_token)

            # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            # get score
            log_score = dit(x_perturb, sigma, condition)

            # calculate loss 
            loss = score_entropy_loss(log_score, sigma.unsqueeze(-1), x_perturb, x, mask_token)
            loss = (dsigma.unsqueeze(-1) * loss).sum(dim=-1).mean()

            loss.backward()
            # gradient cliping - helps to prevent unnecessary divergence 
            torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=1.0)
            # gradient step
            dit_optimizer.step()
            dit_optimizer.zero_grad()

            results_dit_loss = ema(results_dit_loss, loss.item())

            pbar.update(1)
            pbar.set_description('mode:{} loss: {:.2f}'.format(sleep_mode, loss.item()))
            fsq.train()


        # sample
        if (train_step+1) % sampling_freq == 0: ## sample 

            if sleep_mode == 1: # sleep mode - eval FSQ 

                # wake mode visualization shifted to sleep mode 
                if len(wake_visualize_dataset) > 0: # wake mode - visualize wake dataset 
                    idx = np.arange(len(wake_visualize_dataset))
                    np.random.shuffle(idx)
                    x, x_r, cap = wake_visualize_dataset[idx[0]]
                    cap = cap.split('/')
                    cap = ''.join(cap)
                    x, x_r = x.unsqueeze(0), x_r.unsqueeze(0)
                    x = to_img(x)
                    x_r = to_img(x_r)
                    save_path = results_dir + 'trainStep=' + str(train_step) + '_sleepMode=' + str(0) + '_caption=' + cap + '.png'
                    save_img_reconstructed(x[0], x_r[0], save_path)
                    wake_visualize_dataset = []

                # convert patch sequence to img 
                recon_imgs = patch_seq_to_img(recon_x, patch_size, img_channels)

                x_r = to_img(recon_imgs.data)
                x = to_img(imgs.data)

                save_path = results_dir + 'trainStep=' + str(train_step) + '_sleepMode=' + str(sleep_mode) + '_reconstructed.png'
                save_img_reconstructed(x[0], x_r[0], save_path)

            
            if sleep_mode == 2: # dream mode - eval DIT 
            
                # put model in eval mode to avoid dropout
                dit.eval()

                with torch.no_grad():

                    # visualize fantasy 
                    if len(fantasy_dataset) > 0:
                        idx = np.arange(len(fantasy_dataset))
                        np.random.shuffle(idx)
                        x, cap = fantasy_dataset[idx[0]]
                        cap = cap.split('/')
                        cap = ''.join(cap)
                        x = x.unsqueeze(0)
                        x = to_img(x)
                        save_path = results_dir + 'trainStep=' + str(train_step) + '_sleepMode=' + str(sleep_mode + 0.5) + '_caption=' + cap + '.png'
                        save_img_generated(x[0], save_path)

                    # fetch the caption from the minibatch corresponding to sample_caption_emb
                    sample_caption_string = minibatch[0][1]
                    sample_caption_string = sample_caption_string.split('/')
                    sample_caption_string = ''.join(sample_caption_string)

                    # get sample tokens corresponding to indices of codebook 
                    x_sample = get_sample(dit, seq_len, mask_token, vocab_size, num_sampling_steps, dream_visualize_sample_batch_size, sample_caption_emb, cfg_scale, device) # x_sample.shape: [b, seqlen]
                    x_sample = x_sample.flatten() # shape: [b * seqlen]

                    # get codebook vectors indexed by x_sample
                    sampled_img_latents = fsq.codebook[x_sample] # sampled_img_latents.shape: [b, seqlen, latent_dim]
                    gen_img_patch_seq, _ = fsq.decode(sampled_img_latents.float())

                    # convert patch sequence to img 
                    gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

                    # save generated img
                    gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0,1)
                    # bgr to rgb 
                    gen_imgs = torch.flip(gen_imgs, dims=(1,))
                    grid = make_grid(gen_imgs, nrow=2)
                    save_image(grid, f"{results_dir}trainStep={train_step}_sleepMode={sleep_mode}_caption={sample_caption_string}.png")

                    # save original img for reference 
                    ori_img = imgs[0] # [c,h,w]
                    ori_img = ori_img.permute(1,2,0) # [h,w,c]
                    ori_img = (ori_img * 0.5 + 0.5).clamp(0,1)

                    # save ori img
                    save_path = results_dir + 'trainStep=' + str(train_step) + '_sleepMode=' + str(sleep_mode) + '_caption=' + sample_caption_string + '_original.png'
                    save_img_generated(ori_img, save_path)

                # put model back to train mode 
                dit.train()


        if (train_step+1) % plot_freq == 0: ## save ckpt and plot losses


            if sleep_mode == 1: # sleep mode - plot for FSQ

                # save ckpt 
                save_ckpt(device, fsq_ckpt_path, fsq, fsq_optimizer)

                # wake mode plots shifted to sleep mode

                fig = plt.figure()
                plt.plot(results_test_items, label='test_items')
                plt.legend()
                plt.title('val:{} clipThresh:{}'.format(results_test_items[-1], clip_score_threshold_wake))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0) + '.png'
                fig.savefig(save_path)

                # fig = plt.figure()
                # plt.plot(results_wake_ps_loss, label='wake_prediction_loss')
                # plt.legend()
                # plt.title('final_val:{:.3f}'.format(results_wake_ps_loss[-1]))
                # plt.ylim([0, 10])
                # save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0 + 0.5) + '.png'
                # fig.savefig(save_path)


                # plot results
                fig, ax = plt.subplots(2,2, figsize=(15,10))

                ax[0,0].plot(results_train_loss, label='train_loss')
                ax[0,0].plot(results_recon_loss, label='recon_loss')
                ax[0,0].plot(results_compress_loss, label='compress_loss')
                ax[0,0].plot(results_sleep_ps_loss, label='sleep_prediction_loss')
                ax[0,0].plot(results_sleep_fantasy_loss, label='sleep_fantasy_loss')
                ax[0,0].legend()
                ax[0,0].set(xlabel='eval_iters')
                ax[0,0].set_title('train:{:.3f} recon:{:.3f} compress:{:.3f} pred:{:.3f} fantasy:{:.3f}'.format(results_train_loss[-1], results_recon_loss[-1], results_compress_loss[-1], results_sleep_ps_loss[-1], results_sleep_fantasy_loss[-1]))

                ax[1,0].plot(results_codebook_unique, label='codebook_unique')
                ax[1,0].legend()
                ax[1,0].set(xlabel='eval_iters')
                ax[1,0].set_title('val:{:.3f}'.format(results_codebook_unique[-1]))

                ax[0,1].plot(results_codebook_usage, label='codebook_usage')
                ax[0,1].legend()
                ax[0,1].set(xlabel='train_iters')
                ax[0,1].set_title('val:{:.3f}'.format(results_codebook_usage[-1]))

                ax[1,1].plot(results_masked_latents, label='percent_masked_latents')
                ax[1,1].legend()
                ax[1,1].set(xlabel='train_iters')
                ax[1,1].set_title('val:{:.3f}'.format(results_masked_latents[-1]))

                # plt.suptitle('final_train_loss: ' + str(results_train_loss[-1]))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(sleep_mode) + '.png'
                plt.savefig(save_path)

                if first_plot[sleep_mode]:
                    results_train_loss, results_recon_loss, results_compress_loss, results_sleep_ps_loss, results_sleep_fantasy_loss = [], [], [], [], []
                    first_plot[sleep_mode] = False  


            if sleep_mode == 2: # dream mode - plot for DIT 

                # save ckpt 
                save_ckpt(device, dit_ckpt_path, dit, dit_optimizer)

                # plot num fantasies
                fig = plt.figure()
                plt.plot(results_num_fantasies, label='num_fantasies')
                plt.legend()
                plt.title('val:{} clipThresh:{}'.format(results_num_fantasies[-1], clip_score_threshold_fantasy))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(sleep_mode + 0.5) + '.png'
                fig.savefig(save_path)

                # plot dit loss
                fig = plt.figure()
                plt.plot(results_dit_loss, label='dit_loss')
                plt.legend()
                plt.title('final_loss:{:.3f}'.format(results_dit_loss[-1]))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(sleep_mode) + '.png'
                fig.savefig(save_path)

                if first_plot[sleep_mode]:
                    result_dit_loss = []
                    first_plot[sleep_mode] = False  


        train_step += 1
        sleep_mode_counter += 1


    pbar.close()
