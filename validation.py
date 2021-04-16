import torch
import torch.nn as nn
import torch.utils.data as data
from data_loader import get_loader
from torchvision import transforms
from models import EncoderCNN, DecoderRNN
import math
import utils
import json
import os

def validate(encoder, decoder, criterion, data_loader, vocab_size, epoch, device='cpu', save_captions=False):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        val_loss = 0

        total_step = len(data_loader.dataset.paths) #number of images in val dataset
        predicted_captions = []
        print("Running Validation...")
        for batch in data_loader:
            # Obtain the batch.
            images, captions, img_id = batch #next(iter(data_loader))
            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1))
            val_loss += loss

            if save_captions:
                pred = decoder.sample(features.unsqueeze(1))
                caption = utils.clean_sentence(pred, data_loader)
                predicted_captions.append({"image_id": int(img_id), "caption": str(caption)})

        val_loss /= total_step
        print("Validation Loss for epoch " + str(epoch) + ': ' + str(float(val_loss)))

        if save_captions:
            return  val_loss, predicted_captions

        return val_loss

def test(encoder, decoder, data_loader):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        for batch in data_loader:
            # Obtain the batch.
            orig_img, img = batch #next(iter(data_loader))
            # Move batch of images and captions to GPU if CUDA is available.
            img = img.to(device)

            features = encoder(img)

            pred = decoder.sample(features.unsqueeze(1))
            caption = utils.clean_sentence(pred, data_loader)

            #TODO: show image and caption

if __name__ == "__main__":
    # test validation
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    batch_size = 32  # batch size
    vocab_threshold = 6  # minimum word count threshold
    vocab_from_file = False  # if True, load existing vocab file
    embed_size = 512  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder

    val_data_loader = get_loader(transform=transform_train, mode='val')

    vocab_size = len(val_data_loader.dataset.vocab)

    # Initialize the encoder and decoder.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    encoder.eval()
    decoder.eval()

    # Load the trained weights.
    MODEL_DIR = './Models/BatchSizes/1epoch_' + str(batch_size) + 'batch/'
    encoder_path = os.path.join(MODEL_DIR, 'encoder-1epoch-%dbatch.pkl' % batch_size)
    decoder_path = os.path.join(MODEL_DIR, 'decoder-1epoch-%dbatch.pkl' % batch_size)
    encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    _, predicted_captions = validate(encoder, decoder, criterion, val_data_loader, vocab_size, 1, device,
                                     save_captions=True)

    RESULTS_DIR = './results/BatchSizes/1epoch_' + str(batch_size) + 'batch/''

    with open(os.path.join(RESULTS_DIR, 'val_captions.json'), 'w')) as fp:
        json.dump(predicted_captions, fp)