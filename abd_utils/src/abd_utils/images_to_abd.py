import click
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image


@click.command()
@click.option('--images', help='input directory')
@click.option('--output', help='output directory')
def main(images, output):
    """ convert tiled images to abd format for building detection """

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, 'images'), exist_ok=True)
    cover = pd.DataFrame()
    list_tiles = os.listdir(images)
    list_tiles = [x for x in list_tiles if x.endswith(".png")]

    for num, file in enumerate(tqdm(list_tiles)):
        # create new directory
        zoom, x, y, ext = file.split('.')
        os.makedirs(os.path.join(output, 'images', zoom), exist_ok=True)
        os.makedirs(os.path.join(output, 'images', zoom, x), exist_ok=True)
        if not os.path.exists(os.path.join(output, 'images', zoom, x, y + '.tiff')):
            # save as tiff
            im = Image.open(os.path.join(images, file))
            im_resized = im.resize((512, 512))
            im_resized.save(os.path.join(output, 'images', zoom, x, y + '.tiff'))
        # add to cover
        cover = cover.append(pd.Series({'x': x,
                                        'y': y,
                                        'z': zoom}), ignore_index=True)
        if num % 1000 == 0:
            cover.to_csv(os.path.join(output, 'cover.csv'), header=False, index=False)
    # save cover
    cover.to_csv(os.path.join(output, 'cover.csv'), header=False, index=False)


if __name__ == '__main__':
    main()

