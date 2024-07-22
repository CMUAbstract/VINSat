from ultralytics import YOLO


def train(model_path, dataset_yaml_path):
    # Load pretrained yolo model
    model = YOLO(model_path)

    # Train
    results = model.train(
        data=dataset_yaml_path,
        imgsz=1216,
        epochs=100,
        batch=4,
        name="17R_n1342_top_salient_mse",
        degrees=180.0,
        scale=0.5,
        perspective=0.00017,
        lr0=0.00428,
        lrf=0.00933,
        momentum=0.91146,
        weight_decay=0.00042,
        warmup_epochs=3.34695,
        warmup_momentum=0.54918,
        box=6.50223,
        cls=0.65876,
        dfl=1.28939,
        hsv_h=0.00959,
        hsv_s=0.59062,
        hsv_v=0.34087,
        translate=0.14881,
        shear=0.0,
        flipud=0.0,
        fliplr=0.505,
        mosaic=0.92206,
        mixup=0.0,
        copy_paste=0.0,
    )
    return results


if __name__ == "__main__":
    model_path = "models/yolov8s.pt"
    dataset_yaml_path = "datasets/12R.yaml"
    results = train(model_path, dataset_yaml_path)
