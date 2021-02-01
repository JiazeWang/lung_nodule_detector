import os
rootrecord = []
dirrecord = []
for root,dirs,files in os.walk("CT-Lung/"):
    for dir in dirs:
        pathnow = os.path.join(root, dir)
        dirrecord.append(dirs)
        rootrecord.append(root)
root = set(root)
print("root", len(rootrecord))
print("dirs", len(dirrecord))
