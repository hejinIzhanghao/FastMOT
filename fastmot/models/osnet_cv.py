import cv2


class OSNet_CV:
    def __init__(self, config):

        model_cfg = config['model_cfg']
        model_weights = config['model_weights']

        self.input_shape = config['input_shape']
        self.net = cv2.dnn.readNetFromCaffe(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __call__(self, frame):
        return self.extract(frame)

    def extract(self, frame):

        blob = cv2.dnn.blobFromImage(frame, (1.0 / 255), (self.input_shape[0], self.input_shape[1]), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        return out
