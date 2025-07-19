import random
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import image
from PIL import Image
from os import listdir
from os.path import isdir, isfile, join


def load_data(container_path='datasets', folders=['A', 'B', 'C'],
              size=2000, test_split=0.2, seed=0):
    """
    Loads sign language dataset.
    Args:
        container_path: Path to the dataset folder.
        folders: List of folder names for each class.
        size: Total number of images to load (randomly selected).
        test_split: Fraction of data to reserve for testing.
        seed: Random seed for reproducibility.
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """

    filenames, labels = [], []

    for label, folder in enumerate(folders):
        folder_path = join(container_path, folder)
        images = []

        for fname in sorted(listdir(folder_path)):
            file_path = join(folder_path, fname)

            # Skip directories like .ipynb_checkpoints
            if not isfile(file_path):
                continue

            try:
                with Image.open(file_path) as img:
                    images.append(file_path)
            except Exception:
                continue  # Skip unreadable or invalid image files

        labels.extend(len(images) * [label])
        filenames.extend(images)

    # Shuffle and limit data
    random.seed(seed)
    data = list(zip(filenames, labels))
    random.shuffle(data)
    data = data[:size]
    filenames, labels = zip(*data)

    # Get the images and normalize
    x = paths_to_tensor(filenames).astype('float32') / 255.0
    y = np.array(labels)

    # Split into train/test sets
    split_idx = int(len(x) * (1 - test_split))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return (x_train, y_train), (x_test, y_test)


def path_to_tensor(img_path, size):
    """
    Loads a single image and converts it into a 4D tensor.
    """
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, size=50):
    """
    Loads and stacks multiple images into a single 4D tensor.
    """
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)




# import random
# import numpy as np
# from keras.utils import to_categorical
# from keras.preprocessing import image
# from os import listdir
# from os.path import isdir, join


# def load_data(container_path='datasets', folders=['A', 'B', 'C'],
#               size=2000, test_split=0.2, seed=0):
#     """
#     Loads sign language dataset.
#     """
    
#     filenames, labels = [], []

#     for label, folder in enumerate(folders):
#         folder_path = join(container_path, folder)
#         images = [join(folder_path, d) for d in sorted(listdir(folder_path)) if not isdir(join(folder_path, d))]

#         labels.extend(len(images) * [label])
#         filenames.extend(images)
    
#     random.seed(seed)
#     data = list(zip(filenames, labels))
#     random.shuffle(data)
#     data = data[:size]
#     filenames, labels = zip(*data)

    
#     # Get the images
#     x = paths_to_tensor(filenames).astype('float32')/255
#     # Store the one-hot targets
#     y = np.array(labels)

#     x_train = np.array(x[:int(len(x) * (1 - test_split))])
#     y_train = np.array(y[:int(len(x) * (1 - test_split))])
#     x_test = np.array(x[int(len(x) * (1 - test_split)):])
#     y_test = np.array(y[int(len(x) * (1 - test_split)):])

#     return (x_train, y_train), (x_test, y_test)


# def path_to_tensor(img_path, size):
#     # loads RGB image as PIL.Image.Image type
#     img = image.load_img(img_path, target_size=(size, size))
#     # convert PIL.Image.Image type to 3D tensor
#     x = image.img_to_array(img)
#     # convert 3D tensor to 4D tensor 
#     return np.expand_dims(x, axis=0)

# def paths_to_tensor(img_paths, size=50):
#     list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
#     return np.vstack(list_of_tensors)


# """
#     num_types = len(data['target_names'])
#     targets = np_utils.to_categorical(np.array(data['target']), num_types)
# """