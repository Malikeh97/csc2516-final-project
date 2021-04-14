import matplotlib.pyplot as plt

# TODO : not working yet
def clean_sentence(output, data_loader):
    list_string = []

    for idx in output:
        list_string.append(data_loader.dataset.vocab.idx2word[idx])

    list_string = list_string[1:-1]  # Discard <start> and <end> words
    sentence = ' '.join(list_string)  # Convert list of string to full string
    sentence = sentence.capitalize()  # Capitalize the first letter of the first word
    return sentence

def plotLosses(train_loss, val_loss, title):
    plt.figure()
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)

# Plot losses sanity check
# plotLosses([5.8, 4.7, 3.5 ,3.4, 3.2], [5.8 + 1, 4.7 + 0.9, 3.5 + 0.9, 3.4 + 0.7, 3.2 + 0.8], "Cross entropy Loss (per epoch)", 'Epochs')
# plt.show()