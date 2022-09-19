from symbol import parameters
from torch.utils.data import Dataset
from PIL import Image


# load train dataset by load .txt file
class MyDataset(Dataset):
    def __init__(self, txt_img_source, txt_img_target, txt_img_label, transform=None):
        """ train dataset initialize

        Args:
            txt_img_source (string): txt file path of source image
            txt_img_target (string): txt file path of target image
            txt_img_label (string): txt file path of label image
            transform (_type_, optional): transforms in pytorch, pre-process images
        """
        
        # open the txt file
        txt_source = open(txt_img_source, 'r')
        txt_target = open(txt_img_target, 'r')
        txt_label = open(txt_img_label, 'r')

        # prepare lists to store the paths of each images
        images_source = []
        images_target = []
        images_label = []

        # load every line/path into lists
        for line in txt_source:
            line = line.rstrip()  # delete the '\n'
            images_source.append(line)
        txt_source.close()

        for line in txt_target:
            line = line.rstrip()  
            images_target.append(line)
        txt_target.close()

        for line in txt_label:
            line = line.rstrip() 
            images_label.append(line)
        txt_label.close()

        # define the private variable
        self.images_source = images_source
        self.images_target = images_target
        self.images_label = images_label
        self.transform = transform

    def __getitem__(self, index):
        """ Dataloader load data by the function

        Args:
            index (int): index

        Returns:
            images: true data
        """

        # load image from path by Image module
        img_source = Image.open(self.images_source[index]).convert("RGB")
        img_target = Image.open(self.images_target[index]).convert("RGB")
        img_label = Image.open(self.images_label[index]).convert("RGB")

        # pre-process the dataset
        img_source = self.transform(img_source)
        img_target = self.transform(img_target)
        img_label = self.transform(img_label)

        return img_source, img_target, img_label

    def __len__(self):
        """ auto invoke

        Returns:
            number of dataset
        """
        return len(self.images_source)


class TestDataset(Dataset):
    def __init__(self, txt_img_source, txt_img_target, txt_img_label, transform=None):
        txt_source = open(txt_img_source, 'r')
        txt_target = open(txt_img_target, 'r')
        txt_label = open(txt_img_label, 'r')

        images_source = []
        images_target = []
        images_label = []

        for line in txt_source:
            line = line.rstrip()  # 去掉换行
            images_source.append(line)
        txt_source.close()

        for line in txt_target:
            line = line.rstrip()  # 去掉换行
            images_target.append(line)
        txt_target.close()

        for line in txt_label:
            line = line.rstrip()  # 去掉换行
            images_label.append(line)
        txt_label.close()

        self.images_source = images_source
        self.images_target = images_target
        self.images_label = images_label
        self.transform = transform

    def __getitem__(self, index):
        img_source = Image.open(self.images_source[index]).convert("RGB")
        img_target = Image.open(self.images_target[index]).convert("RGB")
        img_label = Image.open(self.images_label[index]).convert("RGB")

        img_source = self.transform(img_source)
        img_target = self.transform(img_target)
        img_label = self.transform(img_label)

        return img_source, img_target, img_label

    def __len__(self):
        return len(self.images_source)
