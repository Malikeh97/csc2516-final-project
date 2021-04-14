import torch
import torch.nn as nn
import torch.utils.data as data
from data_loader import get_loader
from torchvision import transforms
from models import EncoderCNN, DecoderRNN
import math
import utils

def validate(encoder, decoder, criterion, data_loader, vocab_size, device='cpu', save_captions=False):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        val_loss = 0

        total_step = len(data_loader.dataset.paths) #number of images in val dataset

        for batch in data_loader:

            # Obtain the batch.
            images, captions, img_ids = batch #next(iter(data_loader))
            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1))
            val_loss += loss
            print(val_loss)

            # TODO: save captions
            if save_captions:
                pred = decoder.sample(features.unsqueeze(1))
                caption = utils.clean_sentence(pred, data_loader)

        val_loss /= total_step
        return val_loss

if __name__ == "__main__":
    # test validation
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

    val_data_loader = get_loader(transform=transform_train, mode='val')

    vocab_size = len(val_data_loader.dataset.vocab)
    # hard code for testing
    vocab_size = 6293

    # Initialize the encoder and decoder.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    encoder.eval()
    decoder.eval()

    # Load the trained weights.
    encoder.load_state_dict(torch.load('./models/first run/encoder-1.pkl', map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load('./models/first run/decoder-1.pkl', map_location=torch.device('cpu')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)


    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()


    validate(encoder, decoder, criterion, val_data_loader, vocab_size, device, save_captions=True)