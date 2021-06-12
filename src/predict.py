# import
from src.project_parameters import ProjectParameters
from src.model import create_model
from src.utils import get_transform_from_file
from PIL import Image
import torch
from src.data_preparation import MyDataset, collate_fn
from torch.utils.data import DataLoader

# class


class Predict:
    def __init__(self, project_parameters):
        self.project_parameters = project_parameters
        self.model = create_model(project_parameters=project_parameters).eval()
        self.transform = get_transform_from_file(
            filepath=project_parameters.transform_config_path)['predict']
        self.idx_to_class = {v: k for k,
                             v in self.project_parameters.classes.items()}

    def get_result(self, data_path):
        result = []
        if '.png' in data_path or '.jpg' in data_path:
            image = Image.open(fp=data_path).convert('RGB')
            image = self.transform(image)[None, :]
            with torch.no_grad():
                proba = self.model(image)
                result.append(''.join([self.idx_to_class[v]
                                       for v in self.model.ctc_decoder(proba)[0]]))
        else:
            dataset = MyDataset(root=data_path, class_to_idx=self.project_parameters.classes,
                                max_character_length=self.project_parameters.max_character_length, transform=self.transform)
            data_loader = DataLoader(dataset=dataset, batch_size=self.project_parameters.batch_size,
                                     pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)
            with torch.no_grad():
                for image, _, _ in data_loader:
                    proba = self.model(image)
                    for pred in self.model.ctc_decoder(proba):
                        result.append(
                            [''.join([self.idx_to_class[v] for v in pred])][0])
        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict the data path
    result = Predict(project_parameters=project_parameters).get_result(
        data_path=project_parameters.data_path)
    # use [:-1] to remove the latest comma
    # print(('{},'*project_parameters.num_classes).format(*project_parameters.classes.keys())[:-1])
    print(result)
