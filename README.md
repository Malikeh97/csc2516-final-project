# Analysis of Multi-Modal Feature Representations by CLIP for Image Captioning

## Data

All images and their captions were obtained from the [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/) dataset.

## Training

The baseline CNN+RNN model is trained using `train_captioning.py`. The `clip_rnn_image_captioning.ipynb` notebook has modified this code for training the modified CLIP+RNN and CLIP+Attention models.

## Validation

For validation along with the loss function BLEU, CIDEr and other metrics were evaluated on the saved generated captions.
`vizwiz_caption_evaluation.ipynb` uses the evalutation code in `vizwiz_eval_cap` to calculate these. This code was obtained from [here](https://github.com/Yinan-Zhao/vizwiz-caption).

The generated captions and resulting scores are stored in the `validation results/` directory.

## Testing

Testing on all models was preformed using `testing.py` to generate image-caption pairs and TSNE plots of the learned image and word latent spaces. These results are saved in the `test results/` directory.

## Developers ##
- Saad Saleem (University of Toronto, saad@cs.toronto.edu)
- Shamitra Rohan (University of Toronto, shamitra@cs.toronto.edu)
- Malikeh Ehghaghi (University of Toronto, malikeh.ehghaghi@mail.utoronto.ca)

