# import
from os.path import join, basename
from src.project_parameters import ProjectParameters
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from captcha.image import ImageCaptcha
import random
from pytorch_lightning import LightningDataModule
from src.utils import get_transform_from_file
from torch.utils.data.dataset import random_split
import torch

# def


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


# class


class MyDataset(Dataset):
    def __init__(self, root, class_to_idx, max_character_length, transform):
        super().__init__()
        samples = glob(join(root, '*.png'))
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(root)
            raise RuntimeError(msg)
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.max_character_length = max_character_length
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath = self.samples[index]
        label = basename(filepath)[:-4].split('_')[-1]
        target_length = [len(label)]
        label = [self.class_to_idx[v] for v in label]
        image = Image.open(filepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.LongTensor(label), torch.LongTensor(target_length)


class CAPTCHA(Dataset):
    def __init__(self, class_to_idx, max_character_length, num_files, transform):
        super().__init__()
        self.generator = ImageCaptcha()
        self.class_to_idx = class_to_idx
        self.chars = list(self.class_to_idx)[1:]
        self.max_character_length = max_character_length
        self.transform = transform
        self._generate_samples(num_files=num_files)

    def _generate_samples(self, num_files):
        samples = []
        labels = []
        for _ in range(num_files):
            chars = random.choices(self.chars, k=self.max_character_length)
            samples.append(self.generator.generate_image(chars=chars))
            labels.append(chars)
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = self.samples[index]
        label = self.labels[index]
        target_length = [len(label)]
        label = [self.class_to_idx[v] for v in label]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.LongTensor(label), torch.LongTensor(target_length)


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transform_dict = get_transform_from_file(
            filepath=project_parameters.transform_config_path)

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = MyDataset(root=join(self.project_parameters.data_path, stage), class_to_idx=self.project_parameters.classes,
                                                max_character_length=self.project_parameters.max_character_length, transform=self.transform_dict[stage])
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    lengths = (self.project_parameters.max_files, len(
                        self.dataset[stage])-self.project_parameters.max_files)
                    self.dataset[stage] = random_split(
                        dataset=self.dataset[stage], lengths=lengths)[0]
            if self.project_parameters.max_files is not None:
                assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.classes, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset['train'].dataset.class_to_idx, self.project_parameters.classes)
            else:
                assert self.dataset['train'].class_to_idx == self.project_parameters.classes, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset[stage].class_to_idx, self.project_parameters.classes)
        else:
            train_set = CAPTCHA(class_to_idx=self.project_parameters.classes,
                                max_character_length=self.project_parameters.max_character_length, num_files=1000, transform=self.transform_dict['train'])
            val_set = CAPTCHA(class_to_idx=self.project_parameters.classes,
                              max_character_length=self.project_parameters.max_character_length, num_files=250, transform=self.transform_dict['val'])
            test_set = CAPTCHA(class_to_idx=self.project_parameters.classes,
                               max_character_length=self.project_parameters.max_character_length, num_files=500, transform=self.transform_dict['test'])
            # modify the maximum number of files
            for v in [train_set, val_set, test_set]:
                v.samples = v.samples[:self.project_parameters.max_files]
                v.labels = v.labels[:self.project_parameters.max_files]
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)

    def get_data_loaders(self):
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()
