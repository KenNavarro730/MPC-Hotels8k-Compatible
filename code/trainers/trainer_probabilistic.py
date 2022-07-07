from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from criterions.criterion_factory import criterion_factory
from datasets.dataset_factory import dataset_factory
from models.image.image_model_factory import image_model_factory
from models.combiners.modality_combiner_factory import modality_combiner_factory
from models.text.text_model_factory import text_model_factory
from optimizers.optimizer_factory import optimizer_factory
from third_party.pcme.utils.tensor_utils import l2_normalize
from datetime import datetime
import os
import yaml
import torch
import torch.distributions
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pprint
import torch.nn.functional as F
pp = pprint.PrettyPrinter(indent=4)
import neptune.new as neptune
class TrainerProbabilistic:
    def __init__(self, config, path=None):

        self.config = config
        self.train_dataset = dataset_factory(config, split='train') #This will now return hotels50k train
        self.test_dataset = dataset_factory(config, split='test') #returns hotels50k test
        self.train_dataloader = self.train_dataset.get_loader()
        self.test_dataloader = self.test_dataset.get_loader()
        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.image_encoder = image_model_factory(config).cuda()
        self.text_encoder = text_model_factory(config, config.vocab_length).cuda()
        self.modality_combiner = modality_combiner_factory(config).cuda()

        self.criterion = criterion_factory(config.criterion.train.name, config).cuda()
        params = self.get_model_params()

        self.optimizer = optimizer_factory(config, params)
        self.scheduler = StepLR(self.optimizer, step_size=self.config.optimizer.lr_decay_step,
                                gamma=self.config.optimizer.lr_decay_rate)

        if path:
            print("Path given. Loading previous model")
            self.load_models(path)

        self.save_model_path = os.path.join(config.models_root, config.dataset_name, date)
        self.save_config(self.save_model_path)

        print(yaml.dump(self.config))
        self.train_writer = SummaryWriter(os.path.join(self.save_model_path, "train"))
        self.test_writer = SummaryWriter(os.path.join(self.save_model_path, "test"))

    def set_train(self):
        self.image_encoder.train()
        self.text_encoder.train()
        self.modality_combiner.train()
        self.criterion.train()

    def set_eval(self):
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.modality_combiner.eval()
        self.criterion.eval()

    def get_model_params(self):

        params = []
        # low learning rate for pretrained layers on real image datasets
        params.append({
            'params': [p for p in self.image_encoder.encoder.cnn.fc.parameters()],
            'lr': self.config.optimizer.learning_rate
        })
        params.append({
            'params': [p for p in self.image_encoder.encoder.cnn.parameters()],
            'lr': self.config.optimizer.resnet_lr_factor * self.config.optimizer.learning_rate
        })
        params_list = []
        params_list += [param for param in self.image_encoder.parameters()
                        if param.requires_grad]
        params_list += [param for param in self.text_encoder.parameters()
                        if param.requires_grad]
        params_list += [param for param in self.modality_combiner.parameters()
                        if param.requires_grad]
        params_list += [param for param in self.criterion.parameters()
                        if param.requires_grad]
        params.append({'params': params_list})

        for _, p1 in enumerate(params):  # remove duplicated params
            for _, p2 in enumerate(params):
                if p1 is not p2:
                    for p11 in p1['params']:
                        for j, p22 in enumerate(p2['params']):
                            if p11 is p22:
                                p2['params'][j] = torch.tensor(0.0, requires_grad=True)

        return params

    def train(self):
        batch_number = 0
        self.set_train()
        run = neptune.init(
            project="vidarlab/MultiImageClassification",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMTRjODI4OC1iN2U4LTQ5ZWQtODQyNy1hNWU4NzAyOTIzMGYifQ==",
        )
        for epoch in range(self.config.train.num_epochs):
            loss_total = 0
            for image, text, target, text_length in tqdm(self.train_dataloader, desc='Training for epoch ' + str(epoch)):
                proc_image, proc_text, proc_target, text_length = self.prepare_image_data(image), text,\
                                                     self.prepare_image_data(target), self.prepare_text_lens(text_length)
                loss, loss_dict = \
                    self.compute_loss(proc_image, proc_target, text.long().cuda(), text_length)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_total += loss

                batch_number += 1
            run['train/train_loss'].log(loss_total)
            run['current_epoch'].log(epoch)
            cons_and_nep_data, epoch_loss, perfect_acc = self.eval(epoch)
            run['evaluation/validation_loss'].log(epoch_loss)
            for key in cons_and_nep_data:
                run[f'evaluation/recall_@_{key[1:]}'].log(cons_and_nep_data[key])
                print(f'recall_@_{key[1:]}: ', cons_and_nep_data[key])
            run['perfect/recall_1'] = perfect_acc[0]
            run['perfect/recall_5'] = perfect_acc[1]
            run['perfect/recall_10'] = perfect_acc[2]
            run['perfect/recall_50'] = perfect_acc[3]
            self.scheduler.step()
        self.train_writer.close()
        self.test_writer.close()
        run.stop()
    def prepare_image_data(self, data):
        return torch.from_numpy(np.stack(data)).float().cuda()
    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, texts, lens):
        return self.text_encoder(texts, lens)

    def encode(self, modality, data, lens = None):
        if modality == 'image':
            embedding = self.encode_image(data)
        if modality == 'text':
            embedding = self.encode_text(data, lens)
        return embedding

    def compute_loss(self, image, target, text, text_length):

        image_query_embedding = self.encode_image(image)
        text_query_embedding = self.text_encoder(text, text_length)
        target_embeddings = self.encode_image(target)
        source_embeddings = [image_query_embedding, text_query_embedding]
        query_embedding, query_logsigma, z = self.modality_combiner(source_embeddings)
        embeddings = {
            'source': [{
                'mean': embedding['embedding'],
                'logsigma': embedding['logsigma']
            } for embedding in source_embeddings],
            'target': {
                'mean': target_embeddings['embedding'],
                'logsigma': target_embeddings['logsigma']
            },
            'query': {
                'mean': query_embedding,
                'logsigma': query_logsigma
            },
            'query_z': z,
            'target_z': torch.zeros_like(z)
        }


        loss, loss_dict = self.criterion(embeddings)

        return loss, loss_dict

    @torch.no_grad()
    def compute_test_loss(self, image, text, text_length, target):
        self.set_eval()

        image_query_embedding = self.encode_image(image)
        text_query_embedding = self.text_encoder(text, text_length)
        target_embeddings = self.encode_image(target)
        source_embeddings = [image_query_embedding, text_query_embedding]
        query_embedding, query_logsigma, z = self.modality_combiner(source_embeddings)
        embeddings = {
            'source': [{
                'mean': embedding['embedding'],
                'logsigma': embedding['logsigma']
            } for embedding in source_embeddings],
            'target': {
                'mean': target_embeddings['embedding'],
                'logsigma': target_embeddings['logsigma']
            },
            'query': {
                'mean': query_embedding,
                'logsigma': query_logsigma
            },
            'query_z': z,
            'target_z': torch.zeros_like(z)
        }

        loss, loss_dict = self.criterion(embeddings)

        return loss, loss_dict


    def prepare_test_image(self, data):
        return torch.stack(data).float().cuda()

    def prepare_test_texts(self, data, lens):
        text_query = torch.zeros(len(data), max(lens)).long().cuda()
        for i, cap in enumerate(data):
            end = lens[i]
            text_query[i, :end] = cap[:end]
        return text_query

    def prepare_test_data_by_modality(self, modality, data, lens):
        if modality == 'image':
            return self.prepare_test_image(data)
        if modality == 'text':
            return self.prepare_test_texts(data, lens)

    def prepare_text_lens(self, data):
        return torch.from_numpy(np.stack(data)).long().flatten().cuda()

    @torch.no_grad()
    def eval(self, epoch):
        self.set_eval()
        recall_1_sum, recall_5_sum, recall_10_sum, recall_50_sum = 0,0,0,0
        perfect_1_sum, perfect_5_sum, perfect_10_sum, perfect_50_sum = 0,0,0,0
        length = 0
        stackofimages = torch.stack(self.test_dataset.get_test_queries()).float().cuda()
        loss_total = 0
        for image, text, target, text_length in tqdm(self.test_dataloader, desc=f'testing for epoch {epoch}'):
            # Print statements were meant for debugging purposes but are to nice to look at within console.
            database_embeddings = self.encode_image(stackofimages)['embedding']
            assert database_embeddings.shape[0] == self.test_dataset.test_queries_len()
            proc_image = self.prepare_image_data(image)
            proc_target = self.prepare_image_data(target)
            text_len = self.prepare_text_lens(text_length)
            target_indices = []
            loss, _ = self.compute_test_loss(proc_image, text.long().cuda(), text_len, proc_target)
            for target_image in proc_target:
                target_indices.append(torch.where(torch.where(stackofimages == target_image, 1, 0).all(dim=1)==True)[0][0].item())
            image_query_embedding = self.encode_image(proc_image)
            print("text query token shape:", text.shape, " text_length_shape:", text_len.shape)
            text_query_embedding = self.text_encoder(text.long().cuda(), text_len)
            source_embedding = [image_query_embedding, text_query_embedding]
            query_embedding, query_logsigma, z = self.modality_combiner(source_embedding) #Q X Mean Vector Dim where Q equals Query Size
            #query_embedding is the mean vector in this case (you can verify via modality combiner function)
            imgs_f = []
            print("database_embeddings_shape:", database_embeddings.shape) #G x Mean Vector Dim
            for embeddingg in database_embeddings:
                imgs_f += [embeddingg.cpu()]
            imgs_f_tensor = torch.stack(imgs_f)
            normed_imgs = F.normalize(imgs_f_tensor, p=2, dim=-1).cuda()
            normed_query = F.normalize(query_embedding, p=2, dim=-1).cuda()
            print("normed_query_shape:", normed_query.shape," normed_imgs_shape: ", normed_imgs.shape) #Their dimensions should match in at least one index
            similarity_matrix = torch.matmul(normed_query, normed_imgs.transpose(0,-1)) # position (n,d) is similarity between nth query and d'th database image
            del normed_query
            del normed_imgs
            indicess = torch.sort(similarity_matrix,dim=1, descending=True).indices # 16 x Gallery Size
            resultant_k = torch.full((16,), indicess.shape[1]+1)  # Set values at least as high as number of images in gallery + 1
            shortenedindices = indicess[:, :51]
            for index, number in enumerate(target_indices):
                query_row = shortenedindices[index] # 1 x G where G is gallery size
                if number in query_row:
                    resultant_k[index] = torch.where(query_row == number)[0][0].item()
            perfect_test = torch.full((16,), 0)
            recall_1_sum += torch.where(resultant_k < 1)[0].shape[0]
            recall_5_sum += torch.where(resultant_k < 5)[0].shape[0]
            recall_10_sum += torch.where(resultant_k < 10)[0].shape[0]
            recall_50_sum += torch.where(resultant_k < 50)[0].shape[0]
            perfect_1_sum += torch.where(perfect_test<1)[0].shape[0]
            perfect_5_sum += torch.where(perfect_test<5)[0].shape[0]
            perfect_10_sum += torch.where(perfect_test<10)[0].shape[0]
            perfect_50_sum += torch.where(perfect_test<50)[0].shape[0]
            length +=16
            loss_total += loss
        recall_1 = recall_1_sum/length
        recall_5 = recall_5_sum/length
        recall_10 = recall_10_sum/length
        recall_50 = recall_50_sum/length
        perfect_1 = perfect_1_sum/length
        perfect_5 = perfect_5_sum/length
        perfect_10 = perfect_10_sum/length
        perfect_50 = perfect_50_sum/length
        console_and_neptune_data = {'r1':recall_1, 'r5':recall_5, 'r10':recall_10, 'r50':recall_50}
        self.set_train()
        return console_and_neptune_data, loss_total, [perfect_1, perfect_5, perfect_10, perfect_50]

    def save_config(self, save_to):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        path = os.path.join(save_to, 'config.yaml')
        file = open(path, "w")
        yaml.dump(self.config, file)
        file.close()

    def save_models(self, save_to, name):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        state_dict = {
            'image_encoder': self.image_encoder.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'modality_combiner': self.modality_combiner.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state_dict, os.path.join(save_to, name))

    def load_models(self, load_from):

        state_dict = torch.load(load_from)

        self.image_encoder.load_state_dict(state_dict['image_encoder'])
        self.text_encoder.load_state_dict(state_dict['text_encoder'])
        self.modality_combiner.load_state_dict(state_dict['modality_combiner'])
        self.criterion.load_state_dict(state_dict['criterion'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
