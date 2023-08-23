from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import nltk
import os
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import random
import json

def user_loader(img_folders,
               annotations_files,
               transform,
               mode='test',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/home/animesh/Documents/project/opt'):    

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if (mode == 'test') == True:
        img_folder = img_folders
        annotations_file = os.path.join(img_folder,annotations_files)
       

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folderx=img_folders)

    if (mode == 'test') == True:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folderx):
        self.mode = mode
        if (self.mode == 'test') == True:
            self.paths = [annotations_file] #changed
        self.img_folder = img_folderx
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.batch_size = batch_size
        self.transform = transform
                
    def __getitem__(self, index):   # obtain image if in test mode
        if (self.mode != 'train'):
            path = self.paths[0]

            # Convert image to tensor and pre-process using transform
            img_format = 'RGB'
            PIL_image = Image.open(path).convert(img_format)
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image
        # obtain image and caption if in training mode
        elif (self.mode == 'train') == True:
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_format = 'RGB'
            im_id = 'image_id'
            img_id = self.coco.anns[ann_id][im_id]
            f_name = 'file_name'
            path = self.coco.loadImgs(img_id)[0][f_name]

            # Convert image to tensor and pre-process using transform
            image = self.transform(Image.open(os.path.join(self.img_folder, path)).convert(img_format))

            # Convert caption to tensor of word ids.
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            token_itr = [self.vocab(token) for token in nltk.tokenize.word_tokenize(str(caption).lower())]
            caption.extend(token_itr)
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.tensor(caption, dtype=torch.long)

            # return pre-processed image and caption tensors
            return image, caption

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = []
        for i in range(len(self.caption_lengths)):
            if self.caption_lengths[i] == sel_length:
                all_indices.append(i)
        all_indices = np.array(all_indices)[0]
        indices = [np.random.choice(all_indices) for _ in range(self.batch_size)]
        return indices

    def __len__(self):
        if self.mode != 'train':
            return len(self.paths)
        else:
            return len(self.ids)
