from pycocotools.coco import COCO
from collections import Counter
import nltk
import pickle
import os.path

class Vocabulary(object):

    def __init__(self,
        vocab_threshold,
        vocab_file='./vocab.pkl',
        unk_word="<unk>",
        start_word="<start>",
        end_word="<end>",
        annotations_file='/home/animesh/Documents/project/opt/cocoapi/annotations/captions_train2017.json',
        vocab_from_file=False):
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) == False or self.vocab_from_file == False :
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        else:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
           
        
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word("<start>")
        self.add_word("<end>")
        self.add_word("<unk>")
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx = self.idx + 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        i = 0
        while i < len(ids):
            id = ids[i]
            caption = str(coco.anns[id]['caption'])
            caption = caption.lower()
            tokens = nltk.tokenize.word_tokenize(caption)
            counter.update(tokens)
            i += 1


            if (i % 100000 == 0) == True:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = []
        for word, cnt in counter.items():
            if cnt >= self.vocab_threshold:
            	words.append(word)

        i = 0
        while i < len(words):
            word = words[i]
            self.add_word(word)
            i += 1


    def __call__(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_word]
        

    def __len__(self):
        return len(self.word2idx)
