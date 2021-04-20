from data_loader import get_loader
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.manifold import TSNE
import pylab
from models import EncoderCNN, DecoderRNN, DecoderWithAttention
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import utils
import clip
import os
import json

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def tsne_plot(word_embeddings, word2idx, words, image_embeddings, images):
    print("Word Embeddings shape (vocab size, embedding size)", word_embeddings.shape)
    print("Image features shape (num images, embedding size)", image_embeddings.shape)

    mapped_words = TSNE(n_components=2).fit_transform(word_embeddings)
    mapped_words[:, 0] = scale_to_01_range(mapped_words[:, 0])
    mapped_words[:, 1] = scale_to_01_range(mapped_words[:, 1])

    mapped_images = TSNE(n_components=2).fit_transform(image_embeddings.reshape(image_embeddings.shape[0], -1))
    mapped_images[:, 0] = scale_to_01_range(mapped_images[:, 0])
    mapped_images[:, 1] = scale_to_01_range(mapped_images[:, 1])

    plt.figure(figsize=(16, 12))

    width = 4000
    height = 3000
    max_dim = 200
    full_image = Image.new('RGBA', (width, height))
    for i, img in enumerate(images):
        x = mapped_images[i, 0]
        y = mapped_images[i, 1]

        resize = max(1, img.width / max_dim, img.height / max_dim)
        img = img.resize((int(img.width / resize), int(img.height / resize)), Image.ANTIALIAS)
        full_image.paste(img, (int((width - img.width) * x), int((height - img.height) * y)), mask=img.convert('RGBA'))

    plt.imshow(full_image)

    for w in words:
        i = word2idx[w]
        plt.text(mapped_words[i, 0] * width, mapped_words[i, 1] * height, w)

# Needed for Attention model
def get_clip_features(clip_model, x):
    x1 = clip_model.visual.conv1(x) #torch.Size([1, 768, 7, 7])
    x2 = x1.reshape(x1.shape[0], x1.shape[1], -1) #torch.Size([1, 768, 49])
    x3 = x2.permute(0, 2, 1) #torch.Size([1, 768, 49])
    x4 = torch.cat([clip_model.visual.class_embedding.to(x3.dtype) + torch.zeros(x3.shape[0], 1, x3.shape[-1], dtype=x3.dtype, device=x3.device), x3], dim=1)
    #torch.Size([1, 50, 768])
    x5 = x4 + clip_model.visual.positional_embedding.to(x4.dtype) #torch.Size([1, 50, 768])
    x6 = clip_model.visual.ln_pre(x5) #torch.Size([1, 50, 768])
    x7 = x6.permute(1, 0, 2)  # NLD -> LND #torch.Size([50, 1, 768])
    x8 = clip_model.visual.transformer(x7) #torch.Size([50, 1, 768])
    x9 = x8.permute(1, 0, 2)  # LND -> NLD torch.Size([1, 50, 768])
    return x9

# Testing code

# load models
transform_train = transforms.Compose([
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

batch_size = 64            # batch size
vocab_threshold = 6        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder

test_data_loader = get_loader(transform=transform_train, mode='test')

# test word plots
vocab_size = len(test_data_loader.dataset.vocab)
word2idx = test_data_loader.dataset.vocab.word2idx
vocab_size = 6293

attention_dim = 512
dropout = 0.0
alpha_c = 1.
encoder_dim=512
grad_clip = 5.
fine_tune_encoder = True

decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
# decoder = DecoderWithAttention(attention_dim=attention_dim,
#                                embed_dim=embed_size,
#                                decoder_dim=hidden_size,
#                                vocab_size=vocab_size,
#                                dropout=dropout)

decoder.eval()
# decoder.load_state_dict(torch.load('./Models/2LSTMLayers-Final/Epoch3/decoder-2LSTMlayer-final-3.pkl', map_location=torch.device('cpu')))
# decoder.load_state_dict(torch.load('./Models/clip_models/clip_wo_linear_5epoch/decoder-2.pkl', map_location=torch.device('cpu')))


# test image plots
encoder = EncoderCNN(embed_size)
encoder.eval()
# encoder.load_state_dict(torch.load('./Models/2LSTMLayers-Final/Epoch3/encoder-2LSTMlayer-final-3.pkl', map_location=torch.device('cpu')))
clip_model, preprocess = clip.load("ViT-B/32", device=torch.device('cpu'))

i = 0
orig_imgs = []
words_to_draw = []
captions = []
img_embeddings = np.array([])
num_images = 10
for batch in test_data_loader:
    # if i < start_i: continue
    if i >= num_images: break
    orig_img, img = batch

    # convert back to PIL image and save to draw
    pil_img = Image.fromarray(orig_img.squeeze().numpy().astype('uint8'), 'RGB')
    orig_imgs.append(pil_img)

    # features = encoder(img)
    # features = clip_model.encode_image(img)
    features = get_clip_features(clip_model, img)

    if i == 0:
        img_embeddings = features.detach().numpy()
    else:
        img_embeddings = np.vstack((img_embeddings, features.detach().numpy()))

    # Testing attention decoder
    # caplens = torch.tensor([15]).reshape(1, 1)

    # scores, decode_lengths= decoder.sample(features, captions, caplens)

    # scores_copy = scores.clone()
    # _, max_indice = torch.max(scores_copy, dim=2)  # predict the most likely next word, max_indice shape : (1)
    # # print(max_indice)
    #
    # max_indice = max_indice.squeeze()
    # pred = []
    # for t in range(max_indice.shape[0]):
    #     pred.append(max_indice[t].cpu().numpy().item())

    # Testing LSTM decoder
    pred = decoder.sample(features.unsqueeze(1))
    caption = utils.clean_sentence(pred, test_data_loader)
    captions.append(caption)

    # add words to plot
    words_to_draw += caption.lower().split()
    i+=1

# remove repeated words
words_to_draw = list(set(words_to_draw))

SAVE_DIR = './test results/'
with open(os.path.join(SAVE_DIR, 'test_captions.json'), 'w') as fp:
    json.dump(captions, fp)

for i, img in enumerate(orig_imgs):
    img.save(os.path.join(SAVE_DIR, 'images/image_%d.jpeg' % i))

tsne_plot(decoder.embedding.weight.detach().numpy(), word2idx, words_to_draw, img_embeddings, orig_imgs)
plt.show()
