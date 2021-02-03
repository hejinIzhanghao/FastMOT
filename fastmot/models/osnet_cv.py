import numpy as np
import cv2


class OSNet_CV:
    metric = 'euclidean'
    OUTPUT_LAYOUT = 512

    def __init__(self, config):

        model_cfg = config['model_cfg']
        model_weights = config['model_weights']

        self.input_shape = config['input_shape']
        self.net = cv2.dnn.readNetFromCaffe(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __call__(self, frame):
        return self.extract(frame)

    def _extract(self, frame):
        blob = cv2.dnn.blobFromImage(frame, (1.0 / 255), (self.input_shape[0], self.input_shape[1]), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        return out

    def extract(self, frame, detections):
        if len(detections) == 0:
            return np.empty((0, self.OUTPUT_LAYOUT))

        embeddings = []
        for det in detections:
            tlbr_ = det.tlbr.astype(np.int)
            crop = frame[tlbr_[1]:tlbr_[3] + 1, tlbr_[0]:tlbr_[2] + 1]
            feat = self._extract(crop)
            embeddings.append(feat)

        embeddings = np.concatenate(embeddings).reshape(-1, self.OUTPUT_LAYOUT)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
