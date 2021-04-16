from data_loader import get_loader
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.manifold import TSNE
import pylab
from models import EncoderCNN, DecoderRNN
import numpy as np
import tqdm
import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def decoder_tsne_plot_representation(word_embeddings, vocab, num_words=100):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    print(word_embeddings.shape)
    mapped_X = TSNE(n_components=2).fit_transform(word_embeddings.detach().numpy())
    mapped_X[:, 0] = scale_to_01_range(mapped_X[:, 0])
    mapped_X[:, 1] = scale_to_01_range(mapped_X[:, 1])

    pylab.figure(figsize=(12,12))
    for i, w in enumerate(vocab):
        if i == num_words: break
        pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)

    pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
    pylab.show()

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
    plt.show()

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
# vocab_size = len(test_data_loader.dataset.vocab)
# vocab = test_data_loader.dataset.vocab.word2idx
# vocab_size = 6293
#
# decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
# decoder.eval()
# decoder.load_state_dict(torch.load('./models/batch 64/decoder-1.pkl', map_location=torch.device('cpu')))
#
# decoder_tsne_plot_representation(decoder.word_embeddings.weight, vocab, 100)

# test image plots
encoder = EncoderCNN(embed_size)
encoder.eval()
encoder.load_state_dict(torch.load('./models/batch 64/encoder-1.pkl', map_location=torch.device('cpu')))

i = 0
draw_imgs = []
features = np.array([])
num_images = 50
for batch in test_data_loader:
    if i > num_images: break
    orig_img, img = batch

    pil_img = Image.fromarray(orig_img.squeeze().numpy().astype('uint8'), 'RGB')
    draw_imgs.append(pil_img)

    if i == 0:
        features = encoder(img).detach().numpy()
    else:
        features = np.vstack((features, encoder(img).detach().numpy()))
    i+=1

encoder_tsne_plot_representation(features, draw_imgs, num_images)

