from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.images = []
        self.labels = []
        self.split = split  # This defines the split you are going to use
                            # (split files are called 'train.txt' and 'test.txt')
        self.split += ".txt"
        path = root + "/../" + self.split

        self.length = 0
        label_n = 0
        f = open(path, 'r')
        for filename in f:
            if filename[:(filename.find('/'))] != 'BACKGROUND_Google':
                if self.length == 0:
                    label = filename[:(filename.find('/'))]
                else:
                    if label != filename[:(filename.find('/'))]:
                        label = filename[:(filename.find('/'))]
                        label_n += 1
                self.images.append(pil_loader(root + '/' + filename.rstrip()))
                self.labels.append(label_n)
                self.length += 1

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]  # Provide a way to access image and label via index
                                        # Image should be a PIL Image
                                        # label can be int
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = self.length  # Provide a way to get the length (number of elements) of the dataset
        return length
