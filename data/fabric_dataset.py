import os
import xml.etree.cElementTree as ET

import numpy as np

from .util import read_image


class FabricDataset:
    def __init__(self, data_dir, p_id, split):

        if split == 'train':
            id_list_file = os.path.join(
                data_dir, 'ImageSets/Patterns/p%d_train_supervised.txt' % p_id)
        elif split == 'test':
            id_list_file = os.path.join(
                data_dir, 'ImageSets/Patterns/p%d_test.txt' % p_id)
        else:
            raise NotImplementedError

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = FABRIC_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', 'xmls', id_ + '.xml'))

        bbox = list()
        label = list()
        for obj in anno.findall('bbox'):
            ymin = float(obj.find('ymin').text) - 1
            xmin = float(obj.find('xmin').text) - 1
            ymax = float(obj.find('ymax').text) - 1
            xmax = float(obj.find('xmax').text) - 1
            bbox.append([ymin, xmin, ymax, xmax])

            label.append(0)
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'Images', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label

    __getitem__ = get_example


FABRIC_LABEL_NAMES = (
    'defect',
)
