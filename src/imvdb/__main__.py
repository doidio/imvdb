import os
from pathlib import Path

import click
import itk
import pyvista as pv
import pyvista.examples as pve
import numpy as np

import imvdb


@click.group()
def main():
    pass


@main.command()
@click.argument('workspace', type=str)
@click.option('--input_image', type=str, default=None)
@click.option('--iso_value', type=float, default=1500)
@click.option('--threshold_min', type=float, default=1000)
@click.option('--threshold_max', type=float, default=3000)
@click.option('--fog_volume_name', type=str, default='fog_volume')
@click.option('--fog_volume_vdb', type=str, default='fog_volume.vdb')
@click.option('--fog_volume_image', type=str, default='fog_volume.nii.gz')
@click.option('--fog_volume_mesh', type=str, default='fog_volume.stl')
@click.option('--level_set_name', type=str, default='level_set')
@click.option('--level_set_vdb', type=str, default='level_set.vdb')
@click.option('--level_set_image', type=str, default='level_set.nii.gz')
@click.option('--level_set_mesh', type=str, default='level_set.stl')
@click.option('--creator', type=str, default='imvdb')
def demo(workspace: str, input_image: str, iso_value: float, threshold_min: float, threshold_max: float,
         fog_volume_vdb: str, fog_volume_name: str, fog_volume_image: str, fog_volume_mesh: str,
         level_set_vdb: str, level_set_name: str, level_set_image: str, level_set_mesh: str,
         creator: str, ):
    workspace = Path(workspace)
    os.makedirs(workspace, exist_ok=True)

    fog_volume_vdb = workspace / fog_volume_vdb
    fog_volume_image = workspace / fog_volume_image
    fog_volume_mesh = workspace / fog_volume_mesh
    level_set_vdb = workspace / level_set_vdb
    level_set_image = workspace / level_set_image
    level_set_mesh = workspace / level_set_mesh

    if input_image is None:
        # download input image
        ds = pve.download_head_2()
        origin = np.array(ds.origin)
        spacing = np.array(ds.spacing)

        _ = ds.point_data.keys()[0]
        array = ds.point_data[_].reshape(ds.dimensions, order='F')
        array = np.ascontiguousarray(array)[:, ::-1, ::-1].copy().astype(np.float32)
    else:
        # read input image
        array, origin, spacing = imvdb.array_from_imread(input_image)

    iso_value = (iso_value - threshold_min) / (threshold_max - threshold_min)
    array = imvdb.array_normalized(array, (threshold_min, threshold_max))

    # fog volume vdb
    fog_volume = imvdb.fog_volume_from_array(array, origin, spacing)
    fog_volume.name = fog_volume_name
    fog_volume.creator = creator
    imvdb.write(fog_volume, fog_volume_vdb.as_posix())

    # fog volume image
    array, origin, spacing = imvdb.array_from_grid(fog_volume)
    itk_image = itk.image_from_array(array)
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)
    itk.imwrite(itk_image, fog_volume_image.as_posix())

    # fog volume mesh
    grid = pv.UniformGrid(itk.vtk_image_from_image(itk_image))
    mesh = grid.contour(1, None, True, False, False, [iso_value, iso_value], 'point', 'flying_edges')
    mesh.save(fog_volume_mesh.as_posix())

    # level set vdb
    level_set = imvdb.fog_to_sdf(fog_volume, iso_value)
    level_set.name = level_set_name
    level_set.creator = creator
    imvdb.write(level_set, level_set_vdb.as_posix())

    # level set image
    array, origin, spacing = imvdb.array_from_grid(level_set)
    itk_image = itk.image_from_array(array)
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)
    itk.imwrite(itk_image, level_set_image.as_posix())

    # level set mesh
    grid = pv.UniformGrid(itk.vtk_image_from_image(itk_image))
    mesh = grid.contour(1, None, True, False, False, [0, 0], 'point', 'flying_edges')
    mesh.save(level_set_mesh.as_posix())

    # plot
    pl = pv.Plotter(shape=(2, 2))
    camera = pv.Camera()

    pl.subplot(0, 0)
    pl.add_axes()
    pl.add_text('Fog Volume Image', 'upper_edge', 9)
    pl.camera = camera
    pl.camera_position = 'xz'
    read = pv.get_reader(fog_volume_image.as_posix()).read()
    pl.add_volume(read, clim=[-1, 1], opacity='linear', cmap='magma', show_scalar_bar=False)
    pl.reset_camera()

    pl.subplot(0, 1)
    pl.add_axes()
    pl.add_text('Level Set Image', 'upper_edge', 9)
    pl.camera = camera
    read = pv.get_reader(level_set_image.as_posix()).read()
    pl.add_volume(read, clim=[-1, 1], opacity='linear_r', cmap='magma_r', show_scalar_bar=False)
    pl.reset_camera()

    pl.subplot(1, 0)
    pl.add_axes()
    pl.add_text('Fog Volume Mesh', 'upper_edge', 9)
    pl.camera = camera
    pl.camera_position = 'xz'
    pl.add_mesh(pv.get_reader(fog_volume_mesh.as_posix()).read())
    pl.reset_camera()

    pl.subplot(1, 1)
    pl.add_axes()
    pl.add_text('Level Set Mesh', 'upper_edge', 9)
    pl.camera = camera
    pl.add_mesh(pv.get_reader(level_set_mesh.as_posix()).read())
    pl.reset_camera()

    pl.show()


if __name__ == '__main__':
    main()
