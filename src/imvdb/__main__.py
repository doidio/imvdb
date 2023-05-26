import click
import itk
import numpy as np

import imvdb


@click.group()
def main():
    pass


@main.command()
@click.argument('input_image', type=str)
@click.argument('iso_value', type=float)
@click.option('--threshold_min', type=float, default=-np.inf)
@click.option('--threshold_max', type=float, default=np.inf)
@click.option('--fog_volume_vdb', type=str, default=None)
@click.option('--fog_volume_name', type=str, default='fog_volume')
@click.option('--fog_volume_image', type=str, default=None)
@click.option('--level_set_vdb', type=str, default=None)
@click.option('--level_set_name', type=str, default='level_set')
@click.option('--level_set_image', type=str, default=None)
@click.option('--creator', type=str, default='creator')
def image(input_image: str, iso_value: float, threshold_min: float = -np.inf, threshold_max: float = np.inf,
          fog_volume_vdb: str = None, fog_volume_name: str = 'fog_volume', fog_volume_image: str = None,
          level_set_vdb: str = None, level_set_name: str = 'level_set', level_set_image: str = None,
          creator: str = 'creator'):
    array, origin, spacing = imvdb.array_from_imread(input_image, (threshold_min, threshold_max))

    if fog_volume_vdb is not None or level_set_vdb is not None:
        fog_volume = imvdb.fog_volume_from_array(array, origin, spacing)
        fog_volume.name = fog_volume_name
        fog_volume.creator = creator

        if fog_volume_vdb is not None:
            imvdb.write(fog_volume, fog_volume_vdb)

        if fog_volume_image is not None:
            array, origin, spacing = imvdb.array_from_grid(fog_volume)
            itk_image = itk.image_from_array(array)
            itk_image.SetOrigin(origin)
            itk_image.SetSpacing(spacing)
            itk.imwrite(itk_image, fog_volume_image)

        if level_set_vdb is not None:
            iso_value = (iso_value - threshold_min) / (threshold_max - threshold_min)
            level_set = imvdb.fog_to_sdf(fog_volume, iso_value)
            level_set.name = level_set_name
            level_set.creator = creator
            imvdb.write(level_set, level_set_vdb)

            if level_set_image is not None:
                array, origin, spacing = imvdb.array_from_grid(level_set)
                itk_image = itk.image_from_array(array)
                itk_image.SetOrigin(origin)
                itk_image.SetSpacing(spacing)
                itk.imwrite(itk_image, level_set_image)


if __name__ == '__main__':
    main()
