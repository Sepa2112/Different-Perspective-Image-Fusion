import os

# set the train dataset path
train_dataset_path = "../samples/image"

# load all classes
classes = os.listdir(train_dataset_path)
print(classes)

# create train.txt
train_txt = open("train.txt", 'w')

# cycle every classes
for i in range(len(classes)):

    # path of current class
    current_class_path = os.path.join(train_dataset_path, classes[i])

    # all images' names of current class
    current_class_images = os.listdir(current_class_path)

    # cycle to get every image name of current class
    for j in range(len(current_class_images)):
        current_image_path = os.path.join(current_class_path, current_class_images[j])

        # if-else: avoid add a blank at the end of file
        # wirte every image path into the txt
        if i == len(classes) - 1 and j == len(current_class_images) - 1:
            train_txt.write(current_image_path)
        else:
            train_txt.write(current_image_path + '\n')
        print(current_image_path)

train_txt.close()