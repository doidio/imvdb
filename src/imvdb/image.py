import itk
import numpy as np


def _normalize(v, r):
    return (v - r[0]) / (r[1] - r[0])


def array_from_imread(input_image: str) -> (np.ndarray, np.ndarray, np.ndarray):
    image = itk.imread(input_image)
    origin = np.array(itk.origin(image))
    spacing = np.array(itk.spacing(image))
    array = itk.array_from_image(image)
    array = np.swapaxes(array, 0, 2).copy().astype(np.float32)
    return array, origin, spacing


def array_normalized(array: np.ndarray, minmax=(-np.inf, np.inf)) -> np.ndarray:
    if array.ndim == 1:
        array = array[:, np.newaxis, np.newaxis]
    elif array.ndim == 2:
        array = array[:, np.newaxis]
    elif array.ndim > 3:
        raise RuntimeError(f'input array has {array.ndim} > 3 dimensions')

    minmax = (np.max([minmax[0], np.min(array)]), np.min([minmax[1], np.max(array)]))
    array = _normalize(array, minmax)
    array[np.where(array < 0)] = 0
    array[np.where(array > 1)] = 1

    return array
