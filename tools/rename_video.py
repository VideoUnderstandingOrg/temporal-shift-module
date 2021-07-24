import os
import shutil


root_dir = "/workdir/datasets/video_classify/videos"

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if not file.endswith(".mp4"):
            continue
        
        filename = os.path.join(root, file)
        print(filename)
        out_filename = filename.replace(" ", "-")
        shutil.move(filename, out_filename)
