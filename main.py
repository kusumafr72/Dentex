import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

CLASSES = ['Caries','Deep Caries' ]

url = 'https://drive.google.com/uc?id=1-p1fjNoIEsnFTRkveB0kuxqV6ynSFM6P'
output = 'model_final.pth'
gdown.download(url, output, quiet=False)
def load_cfg():
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file('versiobn_3/Toothsegment_3_config.yml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = output
    
    return cfg


def inference(predictor, img):
    return predictor(img)


def visualize_output(cfg, img, outputs):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get('dentex_test'), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('kk', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)


def discriminate(outputs):
    pred_classes = np.array(outputs['instances'].pred_classes)
    mask = np.isin(pred_classes, CLASSES)
    idx = np.nonzero(mask)
    
    # Get Instance values as a dict and leave only the desired ones
    out_fields = outputs['instances'].get_fields()
    for field in out_fields:
        out_fields[field] = out_fields[field][idx]

   return outputs


def main():
    img = cv2.imread('img.png')
    #img = cv2.imread('dog.jpg')
    cv2.imshow('kk', img)
    cv2.waitKey(0)
    cfg = load_cfg()
    predictor = DefaultPredictor(cfg)
    outputs = inference(predictor, img)
    # aaa = outputs.copy()
    # bbb = discriminate(outputs)
    visualize_output(cfg, img, outputs)


if __name__ == "__main__":
    main()
