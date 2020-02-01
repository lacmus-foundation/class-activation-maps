"""Utils for ClassificationPipeline notebooks."""

from IPython import display as ipy_display
import cv2
import lxml.etree
import numpy as np

from tensorflow.keras import (
    models as M,
    backend as K,
)
import tensorflow as tf


def compose(*layers):
    layers = list(layers)
    ret = input_layer = layers.pop(0)
    for i in layers:
        ret = i(ret)
    return M.Model(input_layer, ret)


def get_boxes(e):
    """Get bounding boxes from VOC annotation.

    Extracts bndbox elements text from ElementTree of parsed VOC
    annotations XML file.

    Ignores the class labels (only single class is supported).

    Args:
        e (lxml.etree.ElementTree): ElementTree object to extract bndbox
            elements from.

    Returns:
        List of (x0,y0,x1,y1) tuples.

    """
    return [
        [
            int(i.xpath("bndbox/%s/text()" % j)[0])
            for j in ["xmin", "ymin", "xmax", "ymax"]
        ]
        for i in e.xpath("//object")
    ]


def load_dataset(dataset_dir):
    """Process VOC dataset information.

    Extracts the filename and bounding boxes info for
    each annotation in VOC dataset.

    Ignores the class labels (only single class is supported).

    Args:
        dataset_dir (pathlib.Path): directory where the dataset is
            located

    Returns:
        List of (filename, list_of_bboxes) pairs.

    """
    ret = []
    for i in (dataset_dir / "Annotations").iterdir():
        path = dataset_dir / "JPEGImages" / (i.stem + ".jpg")
        xml_element_tree = lxml.etree.parse(str(i))
        boxes = get_boxes(xml_element_tree)
        ret.append((path, boxes))
    return ret


def get_mask(w, h, bboxes):
    """Make the mask of specified size for the bounding boxes list.

    Args:
        w (int): width of mask in pixels
        h (int): height of mask in pixels
        bboxes (list(tuple)): list of (x0,y0,x1,y0) tuples

    Returns:
        numpy.array: UInt8 array of shape (h,w) with values equal to the
            number of bboxes covering each of its cells.

    """
    mask = np.zeros((h, w), dtype="uint8")
    for x0, y0, x1, y1 in bboxes:
        mask[y0:y1, x0:x1] += 1
    return mask


def display_jpeg(data, bgr=False, quality=95):
    """Display numpy.array as JPEG image in Jupyter Notebook.

    Args:
        data (numpy.array): image data
        bgr (bool, default=False): set to True if image was loaded via
            cv2.imread (i.e. BGR channels order)
        quality (int, default=95): JPEG encoding quality, lower
            value results in image of poor quality and smaller size.
    """
    if not bgr:
        # data should be in BGR for cv2.imencode()
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    _, jpeg_data = cv2.imencode(".jpg", data, [
        cv2.IMWRITE_JPEG_QUALITY, quality])
    ipy_display.display(ipy_display.Image(data=jpeg_data, format="jpg"))


def display_bboxes(filename, bboxes, size=800):
    """Display image with the bboxes on it.

    Args:
        filename (str): path to image file
        bboxes (list): list of (x0,y0,x1,y1) tuples
    """
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    max_dim = max(image.shape[:2])
    scale = size / max_dim
    new_height, new_width = (np.array(image.shape[:2]) * scale).astype("int")
    image = cv2.resize(image, (new_width, new_height), cv2.INTER_MAX)
    bboxes = (np.array(bboxes) * scale).astype("int").tolist()
    for x0, y0, x1, y1 in bboxes:
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    display_jpeg(image, bgr=True)


def drawimgrid(images, rows=2):
    assert len(images) % rows == 0
    images = np.vstack(np.hstack(images[i::rows]) for i in range(rows))
    display_jpeg(images)


def focal_loss(gamma=2., alpha=.25, sparse=False):
    def focal_loss_fixed(y_true, y_pred):
        if sparse:
            y_true = tf.one_hot(tf.squeeze(y_true), 2)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))  # noqa
    return focal_loss_fixed
