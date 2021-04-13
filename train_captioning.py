import nltk
import torch
import torch.nn as nn
from torchvision import transforms
import sys
from data_loader import get_loader
import math
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

batch_size = 32          # batch size
vocab_threshold = 6        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_epochs = 1             # number of training epochs (1 for testing)
save_every = 1             # determines frequency of saving model weights
print_every = 200          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# Build data loader.
data_loader = get_loader(mode='train', batch_size=batch_size, vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)
# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
print(vocab_size)