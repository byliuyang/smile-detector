import numpy as np
from collections import namedtuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.count_nonzero(y == yhat) / len(y)

def measureAccuracyOfPredictors (predictors, X, y):
    pass 

def smile_classifier(face_images, expected_labels):
    max_fPC = 0
    max_row1  = 0
    max_column1 = 0
    max_row2  = 0
    max_column2 = 0

    def predict(face):
        return 1

    Classifier = namedtuple('Classifier', ['row1', 'column1', 'row2', 'column2','predict'])
    return Classifier(row1 = max_row1, column1 = max_column1, row2 = max_row2, column2 = max_column2, predict = predict)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):

    clf = smile_classifier(trainingFaces, trainingLabels)

    predicted_labels = [clf.predict(face) for face in testingFaces]
    print("Accuracy: %f" % fPC(predicted_labels, testingLabels))

    im = testingFaces[0,:,:]
    show_feature(im, clf.column1, clf.row1, clf.column2, clf.row2)

def show_feature(im, column1, row1, column2, row2):
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray', extent=[0,24,24,0])
    
    rect = patches.Rectangle((column1, row1),1,1,linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle((column2, row2),1,1,linewidth=2,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

    plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
