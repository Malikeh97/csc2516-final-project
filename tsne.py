from data_loader import get_loader
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.manifold import TSNE
import pylab
from models import EncoderCNN, DecoderRNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import utils

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def decoder_tsne_plot_representation(word_embeddings, word2idx, words):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    print(word_embeddings.shape)
    mapped_X = TSNE(n_components=2).fit_transform(word_embeddings)
    mapped_X[:, 0] = scale_to_01_range(mapped_X[:, 0])
    mapped_X[:, 1] = scale_to_01_range(mapped_X[:, 1])

    plt.figure(figsize=(12,12))
    for w in words:
        i = word2idx[w]
        plt.text(mapped_X[i, 0], mapped_X[i, 1], w)

    plt.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    plt.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())

def encoder_tsne_plot_representation(features, images, num_images=5):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    print(features.shape)
    mapped_X = TSNE(n_components=2).fit_transform(features)
    mapped_X[:, 0] = scale_to_01_range(mapped_X[:, 0])
    mapped_X[:, 1] = scale_to_01_range(mapped_X[:, 1])

    width = 4000
    height = 3000
    max_dim = 200
    full_image = Image.new('RGBA', (width, height))
    for i, img in enumerate(images):
        if i == num_images: break

        x = mapped_X[i, 0]
        y = mapped_X[i, 1]

        rs = max(1, img.width / max_dim, img.height / max_dim)
        img = img.resize((int(img.width / rs), int(img.height / rs)), Image.ANTIALIAS)
        full_image.paste(img, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=img.convert('RGBA'))

    plt.figure(figsize=(16, 12))
    plt.imshow(full_image)

def tsne_plot(word_embeddings, word2idx, words, image_embeddings, images):
    print("Word Embeddings shape (vocab size, embedding size)", word_embeddings.shape)
    print("Image features shape (num images, embedding size)", image_embeddings.shape)

    mapped_words = TSNE(n_components=2).fit_transform(word_embeddings)
    mapped_words[:, 0] = scale_to_01_range(mapped_words[:, 0])
    mapped_words[:, 1] = scale_to_01_range(mapped_words[:, 1])

    mapped_images = TSNE(n_components=2).fit_transform(image_embeddings)
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

decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()
decoder.load_state_dict(torch.load('./models/batch 64/decoder-1.pkl', map_location=torch.device('cpu')))
#
# decoder_tsne_plot_representation(decoder.word_embeddings.weight.detach().numpy(), vocab, list(word2idx.keys())[0:150])

# test image plots
encoder = EncoderCNN(embed_size)
encoder.eval()
encoder.load_state_dict(torch.load('./models/batch 64/encoder-1.pkl', map_location=torch.device('cpu')))

i = 0
orig_imgs = []
words_to_draw = []
img_embeddings = np.array([])
num_images = 20
for batch in test_data_loader:
    if i >= num_images: break
    orig_img, img = batch

    # convert back to PIL image and save to draw
    pil_img = Image.fromarray(orig_img.squeeze().numpy().astype('uint8'), 'RGB')
    orig_imgs.append(pil_img)

    features = encoder(img)
    if i == 0:
        img_embeddings = features.detach().numpy()
    else:
        img_embeddings = np.vstack((img_embeddings, features.detach().numpy()))

    pred = decoder.sample(features.unsqueeze(1))
    caption = utils.clean_sentence(pred, test_data_loader)

    # add words to plot
    words_to_draw += caption.lower().split()
    i+=1

# remove repeated words
words_to_draw = list(set(words_to_draw))

# decoder_tsne_plot_representation(decoder.word_embeddings.weight.detach().numpy(), word2idx, words_to_draw)
# encoder_tsne_plot_representation(img_embeddings, orig_imgs, num_images)
tsne_plot(decoder.word_embeddings.weight.detach().numpy(), word2idx, words_to_draw, img_embeddings, orig_imgs)
plt.show()
