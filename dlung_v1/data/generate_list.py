import os
for root,dirs,files in os.walk(r"D:\test"):
    for dir in dirs:
        print(dir)
