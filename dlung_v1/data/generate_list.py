import os
rootrecord = []
dirrecord = []
for root,dirs,files in os.walk("CT-Lung/"):
    rootrecord.append(root)
    for dir in dirs:
        pathnow = os.path.join(root, dir)
        dirrecord.append(dirs)
print("root", rootrecord)
print("dirs", dirrecord)
