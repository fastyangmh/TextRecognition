# import
from ruamel.yaml import safe_load
from os.path import isfile
from torchvision import transforms
import numpy as np
import torch

# def


def load_yaml(filepath):
    with open(file=filepath, mode='r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(filepath):
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        transform_dict = {}
        transform_config = load_yaml(filepath=filepath)
        for stage in transform_config.keys():
            transform_dict[stage] = []
            if type(transform_config[stage]) != dict:
                transform_dict[stage] = None
                continue
            for name, value in transform_config[stage].items():
                if value is None:
                    transform_dict[stage].append(
                        eval('transforms.{}()'.format(name)))
                else:
                    if type(value) is dict:
                        value = ('{},'*len(value)).format(*
                                                          ['{}={}'.format(a, b) for a, b in value.items()])
                    transform_dict[stage].append(
                        eval('transforms.{}({})'.format(name, value)))
            transform_dict[stage] = transforms.Compose(transform_dict[stage])
        return transform_dict
    else:
        assert False, 'please check the transform config path: {}'.format(
            filepath)


def load_checkpoint(model, num_classes, use_cuda, checkpoint_path):
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    for k in checkpoint['state_dict'].keys():
        if 'classifier.bias' in k or 'classifier.weight' in k:
            if checkpoint['state_dict'][k].shape[0] != num_classes:
                temp = checkpoint['state_dict'][k]
                checkpoint['state_dict'][k] = torch.stack(
                    [temp.mean(0)]*num_classes, 0)
    if model.loss_function.weight is None:
        # delete the loss_function.weight in the checkpoint, because this key does not work while loading the model.
        if 'loss_function.weight' in checkpoint['state_dict']:
            del checkpoint['state_dict']['loss_function.weight']
    else:
        # assign the new loss_function weight to the checkpoint
        checkpoint['state_dict']['loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model

# class


class CTCDecoder:
    def __init__(self, mode, blank) -> None:
        self.mode = mode
        self.blank = blank

    def _reconstruct(self, labels):
        new_labels = []
        # merge same labels
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # delete blank
        new_labels = [l for l in new_labels if l != self.blank]
        return new_labels

    def greedy_decode(self, emission_log_prob):
        if type(emission_log_prob) != np.ndarray:
            emission_log_prob = emission_log_prob.cpu().data.numpy()
        _, batch_size, _ = emission_log_prob.shape
        labels = []
        for idx in range(batch_size):
            labels.append(self._reconstruct(
                np.argmax(emission_log_prob[:, idx, :], axis=-1)))
        return labels   #list

    def __call__(self, emission_log_prob):
        if self.mode == 'greedy':
            return self.greedy_decode(emission_log_prob=emission_log_prob)
        else:
            pass
