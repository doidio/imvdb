import itk
import numpy as np


def _normalize(v, r):
    return (v - r[0]) / (r[1] - r[0])


def array_from_imread(input_image: str, threshold=(-np.inf, np.inf)):
    image = itk.imread(input_image)
    origin = np.array(itk.origin(image))
    spacing = np.array(itk.spacing(image))

    image = itk.array_from_image(image)
    image = np.swapaxes(image, 0, 2).copy().astype(np.float32)
    if image.ndim == 1:
        image = image[:, np.newaxis, np.newaxis]
    elif image.ndim == 2:
        image = image[:, np.newaxis]
    elif image.ndim > 3:
        raise RuntimeError(f'input image has {image.ndim} > 3 dimensions')

    threshold = (np.max([threshold[0], np.min(image)]), np.min([threshold[1], np.max(image)]))
    image = _normalize(image, threshold)
    image[np.where(image < 0)] = 0
    image[np.where(image > 1)] = 1

    return image, origin, spacing
