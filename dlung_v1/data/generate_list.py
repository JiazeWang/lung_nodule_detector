import os
for root,dirs,files in os.walk("CT-Lung/"):
    for dir in dirs:
        pathnow = os.path.join(root, dir)
        print("path:", pathnow)
        print("files:", files)
