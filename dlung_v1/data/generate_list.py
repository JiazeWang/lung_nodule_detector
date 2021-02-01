import os
for root,dirs,files in os.walk(r"CT-Lung/"):
    for dir in dirs:
        print(dir)
