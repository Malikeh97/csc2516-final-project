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
import cv2

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

# Compute the coordinates of the image on the plot
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

def encoder_tsne_plot_representation(features, images, num_images=5):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    print(features.shape)
    mapped_X = TSNE(n_components=2).fit_transform(features)
    mapped_X[:, 0] = scale_to_01_range(mapped_X[:, 0])
    mapped_X[:, 1] = scale_to_01_range(mapped_X[:, 1])

    tsne_plot = 255 * np.ones((1000, 1000, 3), np.uint8)

    for i, img in enumerate(images):
        if i == num_images: break

        trans = transforms.ToPILImage(mode='RGB')
        img = trans(img.squeeze())
        maxsize = (26, 26)
        img.thumbnail(maxsize)

        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        x = mapped_X[i, 0]
        y = mapped_X[i, 1]
        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(open_cv_image, x, y, 1000, 0)

        tsne_plot[tl_y:br_y, tl_x:br_x, :] = img

    cv2.imshow('t-SNE', tsne_plot)
    cv2.waitKey()

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
#
# decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
# decoder.eval()
# decoder.load_state_dict(torch.load('./models/batch 64/decoder-1.pkl', map_location=torch.device('cpu')))
#
# decoder_tsne_plot_representation(decoder.word_embeddings.weight, vocab)

# test image plots
encoder = EncoderCNN(embed_size)
encoder.eval()
encoder.load_state_dict(torch.load('./models/batch 64/encoder-1.pkl', map_location=torch.device('cpu')))

i = 0
imgs = []
features = np.array([])
for batch in test_data_loader:
    if i > 1: break
    orig_img, img = batch

    imgs.append(img)

    if i == 0:
        features = encoder(img).detach().numpy()
    else:
        features = np.vstack((features, encoder(img).detach().numpy()))
    i+=1

encoder_tsne_plot_representation(features, imgs)

