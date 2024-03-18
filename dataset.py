import numpy as np
import glob
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

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
        self.glob_path = glob.glob(directory)
        self.dataset = []
        self.class_set = []
        # Images follow the pattern of <some random string>_<class>.png, we parse through the string name to extract the number of fingers in the photo.
        for img_path in self.glob_path:
            self.dataset.append(img_path)
            self.class_set.append(int(img_path[-6:-5]))
        # dataset contains paths of all the iamges
        self.dataset = np.array(self.dataset)
        

    def __len__(self):
        # length of list
        return len(self.dataset)

    def __getitem__(self, idx):
        image = read_image(self.dataset[idx], ImageReadMode.UNCHANGED).float()
        label = self.class_set[idx]

        # return image and label
        return(image, label)
    
    




