import os
for root,dirs,files in os.walk("CT-Lung/"):
    for dir in dirs:
        print(os.path.join(root, dir))
