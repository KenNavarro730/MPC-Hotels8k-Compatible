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
        self.path_to_annotations= '/home/tun84049/Downloads/batch_results.csv'
        self.class_train_test_dict = self.create_train_test() #returns a dictionary containing each class and its train and test image files we sample from
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
        self.randomized_caption_to_url = {}
        self.listofcaptions = list(self.caption_to_url)
        random.shuffle(self.listofcaptions)
        for caption in self.listofcaptions:
            self.randomized_caption_to_url[caption] = self.caption_to_url[caption]
        index = 0
        self.train_caption_to_url = {}
        self.test_caption_to_url = {}
        for annotation_key in self.randomized_caption_to_url:
            if index <= 800:
                self.train_caption_to_url[annotation_key] = self.caption_to_url[annotation_key]
            else:
                self.test_caption_to_url[annotation_key] = self.caption_to_url[annotation_key]
            index +=1

        _, self.longest_sequence, self.vocab_set = self.clean_captions(self.randomized_caption_to_url)
        self.train_cleaned_captions, _, _ = self.clean_captions(self.train_caption_to_url)
        self.test_cleaned_captions,_,_ = self.clean_captions(self.test_caption_to_url)
        tokenizer = Tokenizer(num_words=len(self.vocab_set))
        print("Length of vocab set:", len(self.vocab_set))
        tokenizer.fit_on_texts(list(self.vocab_set))
        self.word_to_id_ = tokenizer.word_index
        self.train_samples, self.test_samples, self.database_images = np.array(self._make_dataset())
        random.shuffle(self.train_samples)
        random.shuffle(self.test_samples)
        self.train_text_queries = np.array([s[0] for s in self.train_samples])
        self.train_target_images = np.array([s[1] for s in self.train_samples])
        self.train_image_queries = np.array([s[2] for s in self.train_samples])
        self.train_text_query_length = np.array([s[3] for s in self.train_samples])
        self.test_text_queries = np.array([s[0] for s in self.test_samples])
        self.test_target_images = np.array([s[1] for s in self.test_samples])
        self.test_images_queries = np.array([s[2] for s in self.test_samples])
        self.test_text_query_length = np.array([s[3] for s in self.test_samples])
        self.reset_seed()


    def create_train_test(self):
        path_dict = dict()
        for parent in self.available_parents:
            for classNum in self.available_parents[parent]:
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
    def get_test_queries(self):
        target_images_ = []
        for image_path in self.database_images:
            target_images_.append(self.transform(Image.open(image_path).convert('RGB')))
        print("Len of database images:", len(self.database_images))
        return target_images_
    def test_queries_len(self):
        return len(self.database_images)
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
    def get_train_data(self):
        return self.train_samples
    def get_test_data(self):
        return self.test_samples
    def _make_dataset(self):
        samples = []
        train_data = []
        test_data = []
        database_images = []
        random.shuffle(samples)
        assert len(self.train_cleaned_captions) == len(self.train_caption_to_url)
        assert len(self.test_cleaned_captions) == len(self.test_caption_to_url)
        for train_caption, train_dirty_caption in zip(self.train_cleaned_captions, self.train_caption_to_url):
            train_target_image_path = str(self.caption_to_url[train_dirty_caption])
            parent, classnum = train_target_image_path.split('/')[-4], train_target_image_path.split('/')[-3]
            tokenized_caption, len_of_caption = self.process_captions(train_caption)
            random.shuffle(self.class_train_test_dict[f'{classnum}_train']) #avoids annotations of same class to use same input image querie
            if parent in self.available_parents.keys() and classnum in self.available_parents[parent]:
                for image_url in self.class_train_test_dict[f'{classnum}_train'][:-4]: #up to the last 4 in order to complete the comment above
                    if image_url != train_target_image_path:
                        full_image_url = os.path.join(self.data_directory, str(parent), str(classnum), 'travel_website', image_url)
                        samplea = (tokenized_caption, train_target_image_path, full_image_url, len_of_caption)
                        train_data.append(samplea)
                    if os.path.join(self.data_directory, str(parent), str(classnum), 'travel_website', image_url) not in database_images:
                        database_images.append(os.path.join(self.data_directory, str(parent), str(classnum), 'travel_website', image_url))
            else:
                for image_url_ in self.class_train_test_dict[f'{classnum}_train'][:-4]:
                    if image_url_ != train_target_image_path:
                        full_image_url_ = os.path.join(self.my_path, 'train', str(parent), str(classnum), 'travel_website', image_url_)
                        sampleb = (tokenized_caption, train_target_image_path, full_image_url_ , len_of_caption)
                        train_data.append(sampleb)
                    if os.path.join(self.my_path, 'train', str(parent), str(classnum), 'travel_website', image_url_) not in database_images:
                        database_images.append(os.path.join(self.my_path, 'train', str(parent), str(classnum), 'travel_website', image_url_))
        for test_caption, test_dirty_caption in zip(self.test_cleaned_captions, self.test_caption_to_url):
            test_target_image_path = str(self.caption_to_url[test_dirty_caption])
            tparent, tclassnum = test_target_image_path.split('/')[-4], test_target_image_path.split('/')[-3]
            tokenized_caption_t, len_of_caption_t = self.process_captions(test_caption)
            if tparent in self.available_parents.keys() and tclassnum in self.available_parents[tparent]:
                for image_url_t in self.class_train_test_dict[f'{tclassnum}_test']:
                    if image_url_t != test_target_image_path:
                        full_image_url_t = os.path.join(self.data_directory, str(tparent), str(tclassnum), 'travel_website', image_url_t)
                        samplec = (tokenized_caption_t, test_target_image_path, full_image_url_t, len_of_caption_t)
                        test_data.append(samplec)
                    if os.path.join(self.data_directory, str(tparent), str(tclassnum), 'travel_website',
                                    image_url_t) not in database_images:
                        database_images.append(
                            os.path.join(self.data_directory, str(tparent), str(tclassnum), 'travel_website', image_url_t))
            else:
                for image_url_t_ in self.class_train_test_dict[f'{tclassnum}_test']:
                    if image_url_t_ != test_target_image_path:
                        full_image_url_t_ = os.path.join(self.my_path, 'train', str(tparent), str(tclassnum),
                                                       'travel_website', image_url_t_)
                        sampled = (tokenized_caption_t, test_target_image_path, full_image_url_t_, len_of_caption_t)
                        test_data.append(sampled)
                    if os.path.join(self.my_path, 'train', str(tparent), str(tclassnum), 'travel_website', image_url_t_) not in database_images:
                        database_images.append(os.path.join(self.my_path, 'train', str(tparent), str(tclassnum), 'travel_website',image_url_t_))

        return train_data, test_data, database_images #each element in this list is (text_query, target_image_path, image_query_path)
    def get_loader(self, pin_memory = True):
        if self.split == 'test' or self.split == 'val':
            num_workers = self.config.dataloader.test_num_workers
            persistent_workers=False
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
        image,text,target,text_length = 0,0,0,0
        if self.split == 'train':
            image_path = self.train_image_queries[index]
            text = self.train_text_queries[index]
            target_img_path = self.train_target_images[index]
            target = self.transform(Image.open(target_img_path).convert('RGB'))
            image = self.transform(Image.open(image_path).convert('RGB'))
            text_length = self.train_text_query_length[index]
        if self.split == 'test':
            image_path = self.test_images_queries[index]
            text = self.test_text_queries[index]
            target_img_path = self.test_target_images[index]
            target = self.transform(Image.open(target_img_path).convert('RGB'))
            image = self.transform(Image.open(image_path).convert('RGB'))
            text_length = self.test_text_query_length[index]
        return image, text, target, text_length  #We need to also insert the text query here, this can be done tommorrow once I set everything else up
    def __len__(self):
        if self.split == 'train':
            return len(self.train_samples)
        if self.split == 'test':
            return len(self.test_samples)
