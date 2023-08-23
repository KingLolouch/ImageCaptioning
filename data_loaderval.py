import numpy as np
import nltk
import os
import torch
import torch.utils.data as data
import json
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

def get_loaderval(transform,
               unk_word="<unk>",
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/home/animesh/Documents/project/opt'):


    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'validate':
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2017.json')
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2017/')  

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          unk_word="<unk>",
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word="<start>",
                          end_word="<end>",
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode != 'train':
        data_loader = data.DataLoader(num_workers=num_workers,dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      )

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):

        self.batch_size = batch_size
        self.img_folder = img_folder
        self.mode = mode
        
        if self.mode == 'validate':
            self.coco = COCO(annotation_file = annotations_file)
            print('Obtaining caption lengths...')

            all_tokens = []
            self.ids = list(self.coco.anns.keys())
            
            for index in tqdm(np.arange(len(self.ids))):
                ann_id = self.ids[index]
                annotation = self.coco.anns[ann_id]
                caption = annotation['caption']
                caption_lower = str(caption).lower()
                tokens = nltk.tokenize.word_tokenize(caption_lower)
                all_tokens.append(tokens)
            self.caption_lengths = []
            for token in all_tokens:
                length = len(token)
                self.caption_lengths.append(length)

        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = []
            for item in test_info['images']:
                file_name = item['file_name']
                self.paths.append(file_name)

        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'validate' :
            img_format = 'RGB'
            
            ann_id = self.ids[index]
            img_format = 'RGB'
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            caption = self.coco.anns[ann_id]['caption']
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert(img_format)
            image = self.transform(PIL_image)
            
           
            # Convert caption to tensor of word ids.
            captions = str(caption).lower()
            tokens = nltk.tokenize.word_tokenize(captions)
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            for token in tokens:
                vocab_token = self.vocab(token)
                caption.extend([vocab_token])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            # return pre-processed image and caption tensors
            return img_id, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        indices = []
        for i in np.arange(len(self.caption_lengths)):
            if self.caption_lengths[i] == sel_length:
                indices.append(i)
        all_indices = np.where(indices)[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.ids)

