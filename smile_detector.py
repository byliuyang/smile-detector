import numpy as np
import sys

from collections import namedtuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.count_nonzero(y == yhat) / len(y)

def are_smiling(predictors, images):
    images = np.array(images)
    majority = len(predictors) / 2
    features = [images[:, predictor[0], predictor[1]] > images[:, predictor[2], predictor[3]] for predictor in predictors]
    return np.count_nonzero(np.array(features), axis=0) > majority

def measureAccuracyOfPredictors (predictors, X, y):
    yhat = are_smiling(predictors, X)
    return fPC(y, yhat)

def next_predictor(predictors, im_shape):
    indices = np.indices((im_shape[0], im_shape[1], im_shape[0], im_shape[1])).T
    indices = indices.reshape((im_shape[0] * im_shape[1] * im_shape[0] * im_shape[1], 4))
    return [tuple(index) for index in indices if not ((index[0] == index[2] and index[1] == index[3]) or tuple(index) in predictors)]

def smile_classifier(face_images, expected_labels):
    max_fPC = 0
    max_row1  = 0
    max_column1 = 0
    max_row2  = 0
    max_column2 = 0

    m = 5
    predictors = []

    max_accuracy = 0

    for i in range(m):
        possible_predictors = next_predictor(predictors, face_images[0].shape)
        best_new_predictor = None
        for new_predictor in possible_predictors:
            predictors.append(new_predictor)
            accuracy = measureAccuracyOfPredictors(predictors, face_images, expected_labels)
            if accuracy >= max_accuracy:
                max_accuracy = accuracy
                best_new_predictor = new_predictor
            predictors.pop()
        predictors.append(best_new_predictor)

    def predict(faces):
        return are_smiling(predictors, faces)

    Classifier = namedtuple('Classifier', ['predictors','predict'])
    return Classifier(predictors= predictors, predict= predict)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    clf = smile_classifier(trainingFaces, trainingLabels)

    training_size = (len(trainingFaces) - 1)
    print("Training Size: %d" % (len(trainingFaces) - 1))
    print("Training Accuracy: %f" % fPC(clf.predict(trainingFaces), trainingLabels))
    print("Testing Accuracy: %f" % fPC(clf.predict(testingFaces), testingLabels))

    im = testingFaces[0,:,:]
    visualize_features(im, clf.predictors)

    plt.savefig('example_features_%d.png' % training_size)

def visualize_features(im, predictors):
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray', extent=[0,24,24,0])

    for predictor in predictors:
        rect = patches.Rectangle((predictor[0], predictor[1]),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((predictor[2], predictor[3]),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    if len(sys.argv) != 2:
        print("Usage: python3 smile_detector.py [training set size]")
        exit()
    training_size = int(sys.argv[1])
    stepwiseRegression(trainingFaces[:training_size + 1], trainingLabels[:training_size + 1], testingFaces, testingLabels)
    exit()
