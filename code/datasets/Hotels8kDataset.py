import os
import pickle
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchtext
from torchtext.data import get_tokenizer
import transformers
from transformers import BertTokenizer
def _extract_ids(im_path, target_idx=-2):
    hotel_id = im_path.split(os.sep)[target_idx]
    img_id = im_path.split(os.sep)[-1].split('.')[0]
    return img_id, hotel_id
class Hotels8k(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, seed=2022):
        self.paths = np.load(os.path.join(data_dir, f'{split}.npy'))
        self.seed = seed
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.75, 1.33)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(0.2) #This hyperparameter 0.2 is what model uses
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406])
            ])
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = np.array(self._make_dataset())
        self.image_paths = np.array([s[0] for s in self.samples])
        self.targets = np.array([s[1] for s in self.samples])
        # self.targets2paths = {}
        # for i in range(len(self.samples)):
        #     p = self.image_paths[i]
        #     t = self.targets[i]
        #     if t not in self.targets2paths:
        #         self.targets2paths[t] = [p]
        #     else:
        #         self.targets2paths[t].append(p)
        self.num_classes = len(self.class_to_idx)
        self.train = split == 'train'
        self.reset_seed()
    def reset_seed(self):
        self.rng = np.random.default_rng(self.seed)
    def _find_classes(self):
        classes = set()
        for path in self.paths:
            _, hotel_id = _extract_ids(path, target_idx=-2)
            classes.add(hotel_id)
        classes = list(classes)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    def _make_dataset(self):
        samples = []
        num_missing = 0
        for path in self.paths:
            _, hotel_id = _extract_ids(path, target_idx=-2)
            if hotel_id in self.class_to_idx:
                item = (path, self.class_to_idx[hotel_id])
                samples.append(item)
            else:
                num_missing+=1
        print(num_missing, " hotels missing out of ", len(self.paths))
        return samples
    def caption_to_id(self, data):
        #We need the caption for the image to be seperated by white spaces 'Hello how are you' for example.
        transformer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = get_tokenizer("basic_english")
        words_to_ids = transformer.convert_tokens_to_ids(tokenizer(data))
        return word_to_ids
    def get_loader(self, pin_memory = True):
        if self.split == 'test' or self.split == 'val':
            num_workers = self.config.dataloader.test_num_workers
            persistent_workers= False
        else:
            num_workers = self.config.dataloader.num_workers
            persistent_workers=True
        return torch.utils.data.DataLoader(
            self,
            batch_size = self.config.dataloader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=persistent_workers)
    def __getitem__(self, index):
        path = self.image_paths[index]
        target = torch.from_numpy(self.targets[index]).long()
        
        # chosen_paths = [path]
        # unique_requirement = True
        # poss_paths = self.targets2paths[target]
        # images = torch.stack([self.transform(Image.open(p).convert('RGB')) for p in poss_paths])
        image = self.transform(Image.open(path).convert('RGB'))
        return image, target, index #We need to also insert the text query here, this can be done tommorrow once I set everything else up
    def __len__(self):
        return len(self.samples)