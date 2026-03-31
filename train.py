from ultralytics import YOLO
import config


def train():
    model = YOLO(config.MODEL_NAME)

    model.train(
        data=config.DATA_PATH,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,

        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        close_mosaic=config.CLOSE_MOSAIC,
        scale=config.SCALE,

        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        flipud=config.FLIPUD,
        fliplr=config.FLIPLR,

        project=config.PROJECT_NAME,
        name=config.RUN_NAME,
        exist_ok=config.EXIST_OK
    )


if __name__ == "__main__":
    train()
