import os
import shutil

dataset_path = '/workdir/datasets/video_classify/images'
label_path = '/workdir/datasets/video_classify/labels'
filename_train_output = "train_videofolder.txt"
filename_val_output = 'val_videofolder.txt'

categories = os.listdir(dataset_path)
print("categories: {}".format(categories))

output = []

for class_idx in categories:
    class_path = os.path.join(dataset_path, class_idx)
    for folder in os.listdir(class_path):
        video_frame_path = os.path.join(class_path, folder)
        video_frame_number = len(os.listdir(video_frame_path))
        # video frame path, video frame number, and video groundtruth class
        output.append('%s %d %s'%(video_frame_path, video_frame_number, class_idx))
    
with open(os.path.join(label_path, filename_train_output),'w') as f:
    f.write('\n'.join(output))

shutil.copy(os.path.join(label_path, filename_train_output), os.path.join(label_path, filename_val_output))