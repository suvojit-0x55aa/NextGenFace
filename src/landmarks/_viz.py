import numpy as np
import cv2


def saveLandmarksVerticesProjections(imageTensor, projPoints, landmarks):
    '''
    for debug, render the projected vertices and landmarks on image
    :param images: [w, h, 3]
    :param projPoints: [n, 3]
    :param landmarks: [n, 2]
    :return: tensor [w, h, 3
    '''
    assert(imageTensor.dim() == 3 and imageTensor.shape[-1] == 3 )
    assert(projPoints.dim() == 2 and projPoints.shape[-1] == 2)
    assert(projPoints.shape == landmarks.shape)
    image = imageTensor.clone().detach().cpu().numpy() * 255.
    landmarkCount = landmarks.shape[0]
    for i in range(landmarkCount):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        x = projPoints[i, 0]
        y = projPoints[i, 1]
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    return image
