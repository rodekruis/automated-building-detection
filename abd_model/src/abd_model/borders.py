import numpy as np
from skimage.util import crop

def find_building_intersections(raster_list):
    all_intersections = []
    for i, x in enumerate(raster_list):
        for j, y in enumerate(raster_list[i + 1:]):
            all_intersections.append(x * y)

    # return np.array(all_intersections)
    return np.sum(np.array(all_intersections), axis=0)


def augment_borders(matrix):
    original_matrix = matrix.copy()
    matrix = np.pad(matrix, pad_width=1)
    borders_indices = matrix.nonzero()

    i = borders_indices[0]
    j = borders_indices[1]
    matrix[i + 1, j] = matrix[i + 1, j + 1] = matrix[i, j + 1] = matrix[i - 1, j + 1] = matrix[i - 1, j] = matrix[
        i - 1, j - 1] = matrix[i, j - 1] = matrix[i + 1, j - 1] = 1

    matrix = crop(matrix, crop_width=1)

    return matrix