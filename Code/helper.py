import cv2
import numpy as np
from sklearn.utils import shuffle





def promptForInt(message):
    """
    Prompting for Integer input
    :param message: Informative message when prompting for integer input
    :return: integer input
    """
    result = None

    while result is None:
        try:
            result = int(input(message))
        except ValueError:
            pass
    return result

def createSamples(x, y):
    """
    Returns a list of tuples (x, y)
    :param x: 
    :param y: 
    :return: 
    """
    assert len(x) == len(y)

    return [(x[i], y[i]) for i in range(len(x))]

def generator(samples, batchSize=32, useFlips=False, resize=False):
    """
    Generator to supply batches of sample images and labels
    :param samples: list of sample images file names
    :param batchSize: 
    :param useFlips: adds horizontal flips if True (effectively inflates training set by a factor of 2)
    :param resize: Halves images widths and heights if True
    :return: batch of images and labels
    """
    samplesCount = len(samples)

    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, samplesCount, batchSize):
            batchSamples = samples[offset:offset + batchSize]

            xTrain = []
            yTrain = []
            for batchSample in batchSamples:
                y = float(batchSample[1])

                fileName = batchSample[0]

                image = rgbImage(fileName, resize=resize)

                xTrain.append(image)
                yTrain.append(y)

                if useFlips:
                    flipImg = flipImage(image)
                    xTrain.append(flipImg)
                    yTrain.append(y)

            xTrain = np.array(xTrain)
            yTrain = np.expand_dims(yTrain, axis=1)

            yield shuffle(xTrain, yTrain)  # Since we added flips, better shuffle again




def timeStamp():
    import datetime
    now = datetime.datetime.now()
    y = now.year
    d = now.day
    mo = now.month
    h = now.hour
    m = now.minute
    s = now.second

    return '{}_{}_{}_{}_{}_{}'.format(y, mo, d, h, m, s)



def rgbImage(imageFileName, resize=False):
    """
    Opens image as RGB with OpenCV
    :param imageFileName: 
    :param resize: Halves width and height if True
    :return: RGB image
    """
    image = cv2.imread(imageFileName)
	
    if resize:
        image = cv2.resize(src=image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flipImage(image):
    """
    Horizontal flip with OpenCV
    :param image: 
    :return: Horizontally-flipped image
    """
    return cv2.flip(image, 1)
