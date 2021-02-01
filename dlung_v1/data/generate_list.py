import os
root = []
dirs = []
for root,dirs,files in os.walk("CT-Lung/"):
    root.append(root)
    for dir in dirs:
        pathnow = os.path.join(root, dir)
        dirs.append(dirs))
print("root", root)
print("dirs", dirs)
