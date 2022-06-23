from datasets.coco_dataset import Coco
from datasets.Hotels8kDataset import Hotels8k

def dataset_factory(config, split, **kwargs):
    if config.dataset_name == 'coco':
        return Coco(config, split, kwargs.get('clothing_types', None))
    if config.dataset_name == 'Hotels8k':
        return Hotels8k(config.dataset_path,config, split)

