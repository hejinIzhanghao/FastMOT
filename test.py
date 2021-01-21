#!/usr/bin/env python3

from pathlib import Path
import json
import cv2

import fastmot.utils
import fastmot.models.yolo_cv

import fastmot.models.osnet


def main():
    # load config file
    with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
        config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    detector = fastmot.models.yolo_cv.Yolo_CV(config['size'], config['mot']['yolo_cv'])

    cv2.namedWindow("IMG", cv2.WINDOW_AUTOSIZE)

    frame = cv2.imread("images/00585.png")

    if frame is not None:
        detections = detector.detect(0, frame)

    frame = detector.drawDetection(frame, detections)
    cv2.imshow("IMG", frame)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
