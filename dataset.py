from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class ImageListDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        # load the paths from the text file
        with open(path) as f:
            image_paths = []
            for line in f:
                # good enough for now, make an issue if this fails.
                if line.lower().endswith(('gif', 'jpeg', 'jpg',  'png')):
                    image_paths.append(line.strip())

        self.image_paths = image_paths
        self.transform = transform
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path)
        img = self.transform(img)

        return img
