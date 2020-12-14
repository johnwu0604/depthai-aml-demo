import cv2
import depthai
from pathlib import Path
from helpers import pipeline

class DepthAI():

    def __init__(self):
        self.device = depthai.Device('', False)
        self.p = self.device.create_pipeline(config=pipeline.config)
        if self.p is None:
            raise RuntimeError('Error initializing pipelne')

    def startLoop(self):
        detections = []
        while True:
            nnet_packets, data_packets = self.p.get_available_nnet_and_data_packets()

            for nnet_packet in nnet_packets:
                detections = list(nnet_packet.getDetectedObjects())

            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    data0 = data[0, :, :]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    frame = cv2.merge([data0, data1, data2])

                    img_h = frame.shape[0]
                    img_w = frame.shape[1]

                    for detection in detections:
                        pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                        pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)

                        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

                    cv2.imshow('previewout', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        del p  
        del device
        cv2.destroyAllWindows()

if __name__ == '__main__':
    dai = DepthAI()
    dai.startLoop()
