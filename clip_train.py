# https://github.com/openai/CLIP/issues/83 for the discussion

# Requirements:
# Install CLIP and its dependencies by the following 3 commands
# conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.0
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git

# Download the dataset (note that this is a unique URL that has been generated for the flower dataset)
# !curl -L "https://public.roboflow.com/ds/pmXBn8TqW8?key=ZZm6kgr3sf" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

import os
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import torch.nn as nn
import torch.optim as optim

class image_caption_dataset(Dataset):
    def __init__(self, df):

        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        
        images = transform(Image.open(self.images[idx])) #preprocess from clip.load
        caption = self.caption[idx]
        return images,caption
        

def get_dataset():
    imageList = []
    captionList = []

    for i, cls in enumerate(class_names):
        train_imgs = glob.glob('./train/' + cls + '/*.jpg')
        for img in train_imgs:
            imageList.append(img)
            captionList.append(candidate_captions[i])

    listOfTuples = list(zip(imageList, captionList))
      
    # Converting lists of tuples into pandas Dataframe.
    df = pd.DataFrame(listOfTuples,
                      columns = ['image', 'caption'])

    dataset = image_caption_dataset(df)
    return dataset
    

#https://github.com/openai/CLIP/issues/57
#https://github.com/openai/CLIP/issues/83#issuecomment-817541502
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()
    
if __name__ == "__main__":

    # classes and images for training are stored in folders in the train set
    class_names = os.listdir('./train/')
    class_names.remove('_tokenization.txt')
    
    # Tokenizations are examples, can edit these to improve performance
    # be sure the tokenizations are in the same order as your class_names above!

    candidate_captions = []
    with open('./test/_tokenization.txt') as f:
        candidate_captions = f.read().splitlines()

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    if os.path.exists('./trainedModel.pt'):
        print("loading local model")
        model, transform = clip.load("./trainedModel.pt", device=device, jit=False)  # Must set jit=False for training
    else:
        model, transform = clip.load("ViT-B/32", device=device, jit=False) # Must set jit=False for training

    dataset = get_dataset()

    BATCH_SIZE = 64
    EPOCHS = 1
    train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE) #Define your own dataloader

    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from paper


    for epoch in range(EPOCHS):
        for batch in train_dataloader:
            optimizer.zero_grad()

            list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images

            # images= torch.stack([preprocess(Image.fromarray(img)) for img in list_image],dim=0) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            images = list_image.to(device)
            texts = clip.tokenize(list_txt).to(device)

            logits_per_image, logits_per_text = model(images, texts)
            if device == "cpu":
                ground_truth = torch.arange(len(list_image)).long().to(device)
            else:
                ground_truth = torch.arange(len(list_image)).half().to(device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            
            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
    
    # Save trained model
    torch.save(model, "trainedModel.pt")

