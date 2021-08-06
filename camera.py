import YOLO


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        yolo = YOLO.YOLOv3()
        img = Image.fromarray(frame)
        img = yolo.image(img)
        result = np.asarray(img)

        ret, jpeg = cv2.imencode('.jpg', result)

        return jpeg.tobytes()
