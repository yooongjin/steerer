import os
import glob

path = "/home/cho092871/Desktop/Datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data"

with open(os.path.join(path, "test.txt"), 'w') as f:
    for file in glob.glob(os.path.join(path, 'img', "*.jpg")):
        f.write(file + '\n')