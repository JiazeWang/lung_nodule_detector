import numpy as np
with open ("annotations_three_all.csv", "r") as f:
    lines = f.readlines()[1:]
train = []
for line in lines:
    line = line.rstrip()
    line = line.split(",")
    train.append(line[0])
with open ("val9.csv", "r") as f:
    lines = f.readlines()[1:]
val = []
for line in lines:
    valitem = line.rstrip()
    val.append(valitem)
    if valitem in train:
        train.remove(valitem)
print(len(train))
print(len(val))

train = np.array(train)
val = np.array(val)
np.save("train_new.npy", train)
np.save("val_new.npy", val)