from ultralytics import YOLO


class Detector:
    def __init__(self, smoke_model_path, person_model_path):
        self.smoke_model = YOLO(smoke_model_path)
        self.person_model = YOLO(person_model_path)

    def detect(self, img, smoke_conf, person_conf):
        smoke_results = self.smoke_model(img, conf=smoke_conf, verbose=False)
        person_results = self.person_model(
            img,
            conf=person_conf,
            classes=[0],  # 只检测person
            verbose=False
        )

        return smoke_results, person_results
