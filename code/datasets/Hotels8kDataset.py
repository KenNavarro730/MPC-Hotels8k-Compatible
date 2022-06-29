import os
import pickle
import wordninja
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchtext
from third_party.pcme.datasets._transforms import tokenize
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer
import pandas as pd
import random
def _extract_ids(im_path, target_idx=-2):
    hotel_id = im_path.split(os.sep)[target_idx]
    img_id = im_path.split(os.sep)[-1].split('.')[0]
    return img_id, hotel_id
class Hotels8k(torch.utils.data.Dataset):
    def __init__(self, data_dir, config, split, seed=2022):
        self.my_path = '/shared/data/kenneth_50k'
        self.available_parents = {'-1': ['7100', '12011', '26637', '51516', '52135', '55539'],'17':['38834'], '22':['23228'], '73':['33132'], '74':['1469'], '79':['20771'], '89':['25769']}
        self.unavailable_parents = {'-1': ['11417','35510','53751','66841']}
        self.data_directory = data_dir
        self.seed = seed
        self.split = split
        self.config = config
        self.path_to_annotations= '/home/tun84049/Downloads/batch_results.csv' #replace with path to csv file containing annotations
        self.class_train_test_dict = self.create_train_test(self.available_parents) #returns a dictionary containing each class and its train and test image files we sample from
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.caption_to_url = self.get_url_annotation_pairs(self.path_to_annotations) #Captions are keys, urls are values
        self.cleaned_captions, self.longest_sequence, self.vocab_set = self.clean_captions(self.caption_to_url)
        tokenizer = Tokenizer(num_words=len(self.vocab_set))
        print("Length of vocab set:", len(self.vocab_set))
        tokenizer.fit_on_texts(list(self.vocab_set))
        self.word_to_id_ = tokenizer.word_index
        self.samples = np.array(self._make_dataset())
        random.shuffle(self.samples)
        self.text_queries = np.array([s[0] for s in self.samples])
        self.target_images = np.array([s[1] for s in self.samples])
        self.image_queries = np.array([s[2] for s in self.samples])
        self.text_query_length = np.array([s[3] for s in self.samples])
        self.train = split == 'train'
        self.reset_seed()
    def create_train_test(self, path_dictionary):
        path_dict = dict()
        for parent in path_dictionary:
            for classNum in path_dictionary[parent]:
                listofpaths = os.listdir(os.path.join(self.data_directory, str(parent), str(classNum), 'travel_website/'))
                listofpaths = [value for value in listofpaths if value[-3:] == 'jpg']
                random.shuffle(listofpaths)
                path_dict[f'{classNum}_train'] = listofpaths[:-5]
                path_dict[f'{classNum}_test'] = listofpaths[-5:]
        for missing_parent in self.unavailable_parents:
            for missing_class in self.unavailable_parents[missing_parent]:
                listofpathsv2 = os.listdir(os.path.join(self.my_path, 'train', str(missing_parent), str(missing_class), 'travel_website/'))
                random.shuffle(listofpathsv2)
                path_dict[f'{missing_class}_train'] = listofpathsv2[:-4]
                path_dict[f'{missing_class}_test'] = listofpathsv2[-4:]
        return path_dict
    def get_url_annotation_pairs(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        accepted_data_unique = dataset.loc[dataset['Approve']==1].drop_duplicates(subset='HITId').drop_duplicates(subset='Description')
        caption_to_url = dict()
        setofpaths = set()
        for i in range(len(accepted_data_unique)):
            path = str(accepted_data_unique.iloc[i]['Input.image_path']).replace("/home/tul03156/datasets/hotel_50k_images/", '/shared/data/kenneth_50k/')
            if os.path.exists(path):
                caption_to_url[accepted_data_unique.iloc[i]['Description']] = path
                setofpaths.add(path)
        return caption_to_url
    def clean_captions(self, caption_to_url):
        captionslist = []
        unique_vocab = set()
        for dirty_caption in caption_to_url:
            captionslist.append(str(wordninja.split(str(dirty_caption).replace(',',"").replace("  "," ").replace('"','').replace(';','').replace('.','').replace('-','').replace('#',''))).replace(',','').replace("'", "").replace('[','').replace(']','').replace('``',''))
        maximum = 0
        for caption in captionslist:
            if len(caption) > maximum:
                maximum = len(caption)
            for word in tokenize(caption, caption_drop_prob=0):
                unique_vocab.add(word)
        unique_vocab.add("<start>")
        unique_vocab.add("<end>")
        unique_vocab.add("<unk>")
        return captionslist, maximum, unique_vocab
    def reset_seed(self):
        self.rng = np.random.default_rng(self.seed)
    def process_captions(self, sentence):
        word_to_ids = torch.tensor([self.word_to_id_[word] for word in tokenize(sentence, caption_drop_prob=0.1) if word != "``"])
        padded_token_ids = torch.zeros(1,self.longest_sequence)
        padded_token_ids[0, :len(word_to_ids)] = word_to_ids
        text_length = [torch.tensor(len(word_to_ids))]
        return padded_token_ids.flatten(), text_length
    def _make_dataset(self):
        samples = []
        query_infused_data = []
        random.shuffle(samples)
        for caption, dirty_caption in zip(self.cleaned_captions, self.caption_to_url):
            target_image_path = str(self.caption_to_url[dirty_caption])
            parent, classnum = target_image_path.split('/')[-4], target_image_path.split('/')[-3]
            tokenized_caption, len_of_caption = self.process_captions(caption)
            if parent in self.available_parents.keys() and classnum in self.available_parents[parent]:
                if self.split == 'train':
                    for image_url in self.class_train_test_dict[f'{classnum}_train']:
                        if image_url != target_image_path:
                            full_image_url = os.path.join(self.data_directory, str(parent), str(classnum), 'travel_website', image_url)
                            sample = (tokenized_caption, target_image_path, full_image_url, len_of_caption)
                            query_infused_data.append(sample)
                if self.split == 'test':
                    for image_url_t in self.class_train_test_dict[f'{classnum}_test']:
                        if image_url_t != target_image_path:
                            full_image_url_t = os.path.join(self.data_directory, str(parent), str(classnum),
                                                          'travel_website', image_url_t)
                            sample = (tokenized_caption, target_image_path, full_image_url_t, len_of_caption)
                            query_infused_data.append(sample)
            else:
                if self.split == 'train':
                    for image_url_ in self.class_train_test_dict[f'{classnum}_train']:
                        if image_url_ != target_image_path:
                            full_image_url_ = os.path.join(self.my_path, 'train', str(parent), str(classnum), 'travel_website', image_url_)
                            sample = (tokenized_caption, target_image_path, full_image_url_ , len_of_caption)
                            query_infused_data.append(sample)
                if self.split == 'test':
                    for image_url_t_ in self.class_train_test_dict[f'{classnum}_test']:
                        if image_url_t_ != target_image_path:
                            full_image_url_t_ = os.path.join(self.my_path, 'train', str(parent), str(classnum), 'travel_website',image_url_t_)
                            sample = (tokenized_caption, target_image_path, full_image_url_t_, len_of_caption)
                            query_infused_data.append(sample)
        return query_infused_data #each element in this list is (text_query, target_image_path, image_query_path)
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
        image_path = self.image_queries[index]
        text = self.text_queries[index]
        target_img_path = self.target_images[index]
        target = self.transform(Image.open(target_img_path).convert('RGB'))
        image = self.transform(Image.open(image_path).convert('RGB'))
        text_length = self.text_query_length[index]
        return image, text, target, text_length  #We need to also insert the text query here, this can be done tommorrow once I set everything else up
    def __len__(self):
        return len(self.samples)
