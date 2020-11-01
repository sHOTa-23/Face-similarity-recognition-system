import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA


def fit_to_my_dataset(model, root_dir, batch_size, img_size):
    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # reading data from root_dir directory and applying preprocessing function, imported from keras
    data_generator = img_gen.flow_from_directory(root_dir,
                                                 target_size=(img_size, img_size),
                                                 batch_size=batch_size, class_mode=None, shuffle=False)
    num_images = len(data_generator.filenames)
    num_epochs = int(math.ceil(num_images / batch_size))

    features_list = model.predict_generator(data_generator, num_epochs)
    return features_list, data_generator


def similar_images(indices,filenames):
    plt.figure(figsize=(15, 10), facecolor='white')
    plotnumber = 1
    for index in indices:
        if plotnumber <= len(indices):
            ax = plt.subplot(2, 4, plotnumber)
            plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')
            plotnumber += 1
    plt.tight_layout()
    plt.show()


def image_processing(img_path):
    input_shape = (img_size, img_size, 3)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    test_img_features = model.predict(preprocessed_img, batch_size=1)
    test_img_compressed = pca.transform(test_img_features)
    return test_img_compressed


if __name__ == '__main__':
    img_size = 224
    batch_size = 64
    root_dir = 'No Mask_HD'
    # importing ResNet50 Model, with it's initial weights and without top-fully connected layer
    # because we need to use features for our purposes, not just classes
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')

    # called only ones just to fit our dataset to resnet50 model
    feature_list, datagen = fit_to_my_dataset(model=model, root_dir=root_dir, batch_size=batch_size, img_size=img_size)

    pca = PCA(n_components=80)
    pca.fit(feature_list)
    compressed_features = pca.transform(feature_list)
    filenames = [root_dir + '/' + s for s in datagen.filenames]

    # knn algorithm with k =5
    neighbors_pca_features = NearestNeighbors(n_neighbors=5,
                                              algorithm='ball_tree',
                                              metric='euclidean')
    neighbors_pca_features.fit(compressed_features)

    img_path = 'be31.jpg'
    test_img_compressed = image_processing(img_path)

    distances, indices = neighbors_pca_features.kneighbors(test_img_compressed)
    plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
    plt.xlabel(img_path.split('.')[0] + '_Original Image', fontsize=20)
    plt.show()
    similar_images(indices[0],filenames)
