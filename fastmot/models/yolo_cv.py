#!/usr/bin/env python3

import numpy as np
import cv2 as cv


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class Yolo_CV:
    def __init__(self, size, config):
        self.size = size

        self.class_ids = config['class_ids']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']
        self.nms_thresh = config['nms_thresh']
        self.input_shape = config['input_shape']

        model_cfg = config['model_cfg']
        model_weights = config['model_weights']
        classes_file = config['classes_file']
        self.model_classes = None
        with open(classes_file, 'rt') as f:
            self.model_classes = f.read().rstrip('\n').split('\n')

        self.net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def __call__(self, frame_id, frame):
        return self.detect(frame_id, frame)

    def detect(self, frame_id, frame):
        """
        run the network.
        """
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, (1.0 / 255), (self.input_shape[0], self.input_shape[1]), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self._getOutputsNames(self.net))

        # Remove the bounding boxes with low confidence
        detections = self._postprocess(frame, outs)

        return detections

    # Get the names of the output layers
    def _getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def _postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_thresh:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        detections = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            tlbr = [top, left, top + height, left + width]
            detections.append((tlbr, classIds[i], confidences[i]))
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        return detections

    def drawDetection(self, frame, detections):
        for det in detections:
            self._drawPred(frame, det.label, det.conf, int(det.tlbr[1]), int(det.tlbr[0]), int(det.tlbr[3]), int(det.tlbr[2]))
        return frame

    def _drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.model_classes:
            assert(classId < len(self.model_classes))
            label = '%s:%s' % (self.model_classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
