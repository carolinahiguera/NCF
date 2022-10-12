import numpy as np
import random
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

LEN_BUFFER = 15

class DIGIT_Dataset(Dataset):
    def __init__(self, root_path, data_path, debug=False, transform=None):
        self.root_path = root_path
        self.path = data_path
        self.digits_bg_path = f"{root_path}/ncf/conf/"
        self.image_size = None
        self.seed_is_set = False
        self.channels = 3
        self.N = 0
        self.debug = debug
        self.transform = transform

        files_total = []
        files = list(sorted(glob.glob(self.path+"/*.npz")))
        n = len(files)
        if self.debug:
            idx = int(n*0.1)
            files = files[0:idx]
        files_total.extend(files)
        self.files = files_total

        self.bg_left =  np.load(f"{self.digits_bg_path}/bg_D20479.npy")
        self.bg_right = np.load(f"{self.digits_bg_path}/bg_D20510.npy")
        self.bg_left = cv2.cvtColor(self.bg_left, cv2.COLOR_BGR2RGB)
        self.bg_right = cv2.cvtColor(self.bg_right, cv2.COLOR_BGR2RGB)


    def __len__(self):
        return len(self.files)

    def _process_images(self, index, ):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            digit_left =  data["digit_left"]
            digit_right =  data["digit_right"]
            ee_pose = data["ee_pose"]

            idx_seq = (LEN_BUFFER-1) - np.array([9,7,4,2,0])

            img_sz = digit_left[0].shape

            img_seq = np.zeros((len(idx_seq), img_sz[2], img_sz[0], img_sz[1]*2))

            for i, idx in enumerate(idx_seq):
                img1 = digit_left[idx]
                img2 =  digit_right[idx]

                img1 = cv2.GaussianBlur(img1, (7, 7), 0)
                img2 = cv2.GaussianBlur(img2, (7, 7), 0)

                img1 = exposure.match_histograms(img1, self.bg_left, multichannel=True)
                img2 = exposure.match_histograms(img2, self.bg_right, multichannel=True)

                img = cv2.hconcat([img1, img2])
                img = img / 255.

                if self.transform:
                    img = self.transform(img)
                    img_seq[i] = img
            
            return img_seq, ee_pose[idx_seq]

        except Exception as e:
           print(e)
           img = self._process_images(index=random.randint(0, self.__len__() - 1) )
           return img

    def get_item(self, index):
        in_seq, ee_pose = self._process_images(index)
        out_seq = np.zeros_like(in_seq)
        out_seq[1:] = in_seq[0:len(in_seq)-1]


        in_seq = torch.from_numpy(in_seq).float()
        out_seq = torch.from_numpy(out_seq).float()
        ee_pose = torch.from_numpy(ee_pose).float()

        sample = {  'in_seq': in_seq,
                    'out_seq':out_seq,
                    'ee_pose': ee_pose}
        
        return sample


    def __getitem__(self, index):
        return self.get_item(index)


# if __name__ == '__main__':
#     # path = "/Users/carolina/Dropbox/sync/UW/ResearchProjects/dataset_bottom/"
#     path = "/home/chiguera/Dropbox/sync/UW/ResearchProjects/Pipeline/tacto_sim/collect_traj/data_shapenet/train_all/"

#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.5,),(0.5,))])

#     dataset = DIGIT_Dataset(data_root=path, transform=transform, debug=True)
#     print(f'Final dataset has length {dataset.N}')

#     unnormalize = transforms.Normalize((-0.5 / 0.5), (1.0 / 0.5))
#     fig, axs = plt.subplots(1, 5)
#     for i in range(5):
#         x = dataset.__getitem__(index=np.random.randint(100))
#         x = unnormalize(x)
#         x = x.permute(0,2,3,1)
#         axs[i].imshow(x[0])
#     plt.show()
    