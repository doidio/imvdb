import ast

import numpy as np

from . import vdb  # noqa

Grid = vdb.Grid


def probe(grid: Grid, ijk: list) -> float:
    return vdb.probe(grid, list(ijk))


def fog_volume_from_array(array: np.ndarray, origin: np.ndarray = np.zeros(3), spacing: np.ndarray = np.ones(3),
                          background: float = 0.0, tolerance: float = 0.0) -> Grid:
    grid = vdb.from_array(array, origin.tolist(), spacing.tolist(), background, tolerance)
    grid.grid_class = 'fog volume'
    return grid


def level_set_from_array(array: np.ndarray, origin: np.ndarray = np.zeros(3), spacing: np.ndarray = np.ones(3),
                         background: float = 0.0, tolerance: float = 0.0) -> Grid:
    grid = vdb.from_array(array, origin.tolist(), spacing.tolist(), background, tolerance)
    grid.grid_class = 'level set'
    return grid


def array_from_grid(grid: Grid) -> np.ndarray:
    metadata = metadata_from_grid(grid)
    array = vdb.to_array(grid)
    array = np.swapaxes(array, 0, 2).copy()
    return array, metadata['origin'], metadata['spacing']


def metadata_from_grid(grid: Grid) -> dict:
    literal = grid.metadata
    data = {}
    for key in literal:
        data[key] = literal[key]
        try:
            data[key] = ast.literal_eval(data[key])
        except Exception:  # noqa
            pass
    return data


def write(grids: list | Grid, filename: str) -> None:
    if not isinstance(grids, list):
        grids = [grids]
    return vdb.write(grids, filename)


def fog_to_sdf(grid: Grid, iso_value: float) -> Grid:
    grid = vdb.fog_to_sdf(grid, float(iso_value))
    grid.grid_class = 'level set'
    return grid


def volume_to_mesh(grid: Grid, iso_value: float, adaptivity: float = 0.0) -> (np.ndarray, np.ndarray, np.ndarray):
    return vdb.volume_to_mesh(grid, float(iso_value), float(adaptivity))


def volume_to_quad_mesh(grid: Grid, iso_value: float) -> (np.ndarray, np.ndarray):
    return vdb.volume_to_quad_mesh(grid, float(iso_value))
