#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import cv2

import fastmot.utils
import fastmot


def main():
    # load config file
    with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
        config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    detector = fastmot.models.yolo_cv.Yolo_CV(config['size'], config['mot']['yolo_cv'])
    extractor = fastmot.models.osnet_cv.OSNet_CV(config['mot']['osnet_cv'])

    cv2.namedWindow("IMG", cv2.WINDOW_AUTOSIZE)

    frame = cv2.imread("images/111_ir_Moment_2.jpg")
    frame = np.uint8(255 * np.ones(frame.shape) - frame)

    detections = []
    if frame is not None:
        detections = detector.detect(0, frame)

    frame = detector.drawDetection(frame, detections)
    cv2.imshow("IMG", frame)
    cv2.waitKey(0)

    embeddings = extractor.extract(frame, detections)
    print(embeddings)


if __name__ == '__main__':
    main()
