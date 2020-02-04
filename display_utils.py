from IPython import display as ipy_display
import cv2
import numpy as np


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
