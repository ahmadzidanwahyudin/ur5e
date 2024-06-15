import torch

BATCH_SIZE = 15 # Increase / decrease according to GPU memeory.
RESIZE_TO = 300 # Resize the image for training and transforms.
NUM_EPOCHS = 50 # Number of epochs to train for.
NUM_WORKERS = 3 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = '/home/desktop/file_TA/dataset/picknplace.v5i.voc/train'
# Validation images and XML files directory.
VALID_DIR = '/home/desktop/file_TA/dataset/picknplace.v5i.voc/valid'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'jeruk','pisang', 'roti panjang', 'roti bulat'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs' 