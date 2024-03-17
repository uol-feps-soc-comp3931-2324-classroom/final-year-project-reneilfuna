import numpy as np
import glob
import cv2
from torch.utils.data import Dataset

class Fingers(Dataset):

    # define constructor
    def __init__(self, directory, transform=None):
        '''
        Arguments:
        + directory (string): Directory containing image data
        + transform (callable): optional transform to apply on sample
        '''
        # directory containing the images
        self.directory = directory
        self.transform = transform
        self.path = glob.glob(directory)
        self.dataset = []
        # Images follow the pattern of <some random string>_<class>.png, we parse through the string name to extract the number of fingers in the photo.
        for img_path in self.glob_path:
            img_label = int(img_path[-6:-5])
            img = cv2.imread(img_path)
            self.dataset.append({'image': img, 'label': img_label})
        self.dataset = np.array(self.dataset)

    def __len__(self):
        # length of the dictionary
        return len(self.dataset)

    def __getiem__(self, idx):
        sample = self.dataset[idx]
        return(self.transform(sample['image']), sample['label'])
    
    




