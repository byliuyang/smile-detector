import numpy as np
import sys

from collections import namedtuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def are_smiling(predictors, images):
    images = np.array(images)
    majority = len(predictors) / 2
    features = [images[:, predictor[0], predictor[1]] > images[:, predictor[2], predictor[3]] for predictor in predictors]
    return np.count_nonzero(np.array(features), axis=0) > majority

def measureAccuracyOfPredictors (predictors, X, y):
    yhat = are_smiling(predictors, X)
    return fPC(y, yhat)

def smile_classifier(face_images, expected_labels):
    m = 5
    predictors = []

    max_accuracy = 0

    im_shape = face_images[0].shape
    for i in range(m):
        best_new_predictor = None
        for r1 in range(im_shape[0]):
            for c1 in range(im_shape[1]):
                for r2 in range(im_shape[0]):
                    for c2 in range(im_shape[1]):
                        new_predictor = (r1, c1, r2, c2)
                        if (new_predictor[0] == new_predictor[2] and new_predictor[1] == new_predictor[3]) or (new_predictor in predictors):
                            continue
                        predictors.append(new_predictor)
                        accuracy = measureAccuracyOfPredictors(predictors, face_images, expected_labels)
                        if accuracy > max_accuracy or (best_new_predictor is None and accuracy == max_accuracy):
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

    training_size = len(trainingFaces)
    print("Training Size: %d" % training_size)
    print("Training Accuracy: %f" % fPC(clf.predict(trainingFaces), trainingLabels))
    print("Testing Accuracy: %f" % fPC(clf.predict(testingFaces), testingLabels))

    im = testingFaces[0,:,:]
    visualize_features(im, clf.predictors)

    plt.savefig('example_features_%d.png' % training_size)

def visualize_features(im, predictors):
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray', extent=[0,24,24,0])

    for predictor in predictors:
        rect = patches.Rectangle((predictor[1], predictor[0]),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((predictor[3], predictor[2]),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    if len(sys.argv) > 2:
        print("Usage: python3 smile_detector.py [training set size]")
        print("Usage: python3 smile_detector.py")
    if len(sys.argv) == 2:
        training_size = int(sys.argv[1])
        print("Training the classifier on %d images" % training_size)
        stepwiseRegression(trainingFaces[:training_size], trainingLabels[:training_size], testingFaces, testingLabels)
    else:
        training_sizes = [400, 800, 1200, 1600, 2000]
        for training_size in training_sizes:
            print("Training the classifier on %d images" % training_size)
            stepwiseRegression(trainingFaces[:training_size], trainingLabels[:training_size], testingFaces, testingLabels)
            print()
    
