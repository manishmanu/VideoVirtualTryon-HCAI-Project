import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import random
import pickle
from skimage.transform import resize
import cv2

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")

# def loader(im1, im3, input_frame_size = (3, 256, 448), output_frame_size = (3, 256, 448), data_aug = True):
def loader(im1, im3, output_frame_size = (3, 320, 256), data_aug = True):
    
    # root = os.path.join(root,'sequences',im_path)

    # if data_aug and random.randint(0, 1):
    #     path_pre2 = os.path.join(root,  "im1.png")
    #     path_mid = os.path.join(root,  "im2.png")
    #     path_pre1 = os.path.join(root,  "im3.png")
    # else:
    #     path_pre1 = os.path.join(root,  "im1.png")
    #     path_mid = os.path.join(root,  "im2.png")
    #     path_pre2 = os.path.join(root,  "im3.png")
    _, h, w = output_frame_size

    im1_pre = cv2.resize(cv2.imread(im1), (w, h), interpolation = cv2.INTER_AREA)
    im3_pre = cv2.resize(cv2.imread(im3), (w, h), interpolation = cv2.INTER_AREA)
    # im_mid = imread(path_mid)

    # h_offset = random.choice(range(256 - input_frame_size[1] + 1))
    # w_offset = random.choice(range(448 - input_frame_size[2] + 1))

    # im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    # im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    # im_mid = im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]

    # if data_aug:
    #     if random.randint(0, 1):
    #         im_pre2 = np.fliplr(im_pre2)
    #         im_mid = np.fliplr(im_mid)
    #         im_pre1 = np.fliplr(im_pre1)
    #     if random.randint(0, 1):
    #         im_pre2 = np.flipud(im_pre2)
    #         im_mid = np.flipud(im_mid)
    #         im_pre1 = np.flipud(im_pre1)

    im1_pre = cv2.cvtColor(im1_pre, cv2.COLOR_BGR2RGB)
    im3_pre = cv2.cvtColor(im3_pre, cv2.COLOR_BGR2RGB)
    X0 = np.transpose(im1_pre,(2,0,1))
    X2 = np.transpose(im3_pre, (2, 0, 1))

    # y = np.transpose(im_mid, (2, 0, 1))
    # return X0.astype("float32")/ 255.0, \
    #         X2.astype("float32")/ 255.0,\
    #         y.astype("float32")/ 255.0
    return X0.astype("float32")/255.0, X2.astype("float32")/255.0



class VITON_Data(data.Dataset):
    def __init__(self, train=True, info_folder="/home/vishnu1911/test2/dataset_generated_new/dataset_generated", pkl_folder="/home/vishnu1911/test2/DAIN"):
        # self.info_folder = info_folder
        vid_names = os.listdir(info_folder)
        self.tup = []
        len_vids = int(0.8 * len(vid_names))
        ctr = 0
        if train:
            vid_names = vid_names[:len_vids]
        else:
            vid_names = vid_names[len_vids:]

        for v in vid_names:
            mi_path = os.path.join(info_folder, v, "masked_input_images")
            pkl_file_name = os.path.join(pkl_folder, v + ".pkl")
            out_path = os.path.join(info_folder, v, "input_images")
            pkl_dict = pickle.load( open(pkl_file_name, "rb" ) )
            # read each image and also transform
            for f in os.listdir(mi_path):
                im2 = f.split(".jpg")[0]
                if im2 not in pkl_dict:
                    continue
                im1 = os.path.join(mi_path, f)
                im3 = os.path.join(out_path, f)
                
                self.tup.append([im1, pkl_dict[im2].squeeze(), im3])

        a = 10

    def __getitem__(self, index):
            im1, im2, im3 = self.tup[index]
            im1, im3 = loader(im1, im3)
            return im1, im2, im3

    def __len__(self):
            return len(self.tup)




# training_data = VITON_Data()

# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)


# im1, im2, im3 =next(iter(train_dataloader))

