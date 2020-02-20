import os
import itertools
from tqdm import tqdm

import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import VGGDistance, LordEmbeddings, LordModulation, LordGenerator, LordStage1, \
	LordEncoders, LordStage2
from model.utils import AverageMeter


class Lord:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.embeddings = LordEmbeddings(config)
		self.modulation = LordModulation(config)
		self.generator = LordGenerator(config)
		self.model = LordStage1(self.embeddings, self.modulation, self.generator)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def train(self, imgs, classes, model_dir, tensorboard_dir):
		imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
		class_ids = torch.from_numpy(classes.astype(int))
		img_ids = torch.arange(imgs.shape[0])

		tensor_dataset = TensorDataset(imgs, img_ids, class_ids)
		data_loader = DataLoader(
			tensor_dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.model.init()
		self.model.to(self.device)

		criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)

		optimizer = Adam([
			{
				'params': self.model.generator.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': self.model.modulation.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': self.model.embeddings.parameters(),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		with SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'stage1')) as summary:
			train_loss = AverageMeter()
			for epoch in range(1, self.config['train']['n_epochs'] + 1):
				self.model.train()
				train_loss.reset()

				with tqdm(iterable=data_loader) as pbar:
					for batch in pbar:
						batch_imgs, batch_img_ids, batch_class_ids = (tensor.to(self.device) for tensor in batch)
						generated_imgs, batch_content_codes, batch_class_codes = self.model(batch_img_ids, batch_class_ids)

						optimizer.zero_grad()

						content_penalty = torch.sum(batch_content_codes ** 2, dim=1).mean()
						loss = criterion(generated_imgs, batch_imgs) + self.config['content_decay'] * content_penalty
						loss.backward()

						optimizer.step()
						scheduler.step()

						train_loss.update(loss.item())
						pbar.set_description_str('epoch #{}'.format(epoch))
						pbar.set_postfix(loss=train_loss.avg)

				torch.save(self.model.generator.state_dict(), os.path.join(model_dir, 'generator.pth'))
				torch.save(self.model.embeddings.state_dict(), os.path.join(model_dir, 'embeddings.pth'))
				torch.save(self.model.modulation.state_dict(), os.path.join(model_dir, 'class_modulation.pth'))

				self.model.eval()
				fixed_sample_img = self.evaluate(imgs, img_ids, class_ids, randomized=False)
				random_sample_img = self.evaluate(imgs, img_ids, class_ids, randomized=True)

				summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)
				summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step=epoch)
				summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step=epoch)

	def train_encoders(self, imgs, classes, model_dir, tensorboard_dir):
		imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
		class_ids = torch.from_numpy(classes.astype(int))
		img_ids = torch.arange(imgs.shape[0])

		tensor_dataset = TensorDataset(imgs, img_ids, class_ids)
		data_loader = DataLoader(
			tensor_dataset, batch_size=self.config['train_encoders']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.embeddings = LordEmbeddings(self.config)
		self.modulation = LordModulation(self.config)
		self.encoders = LordEncoders(self.config)
		self.generator = LordGenerator(self.config)
		self.embeddings.load_state_dict(torch.load(os.path.join(model_dir, 'embeddings.pth')))
		self.modulation.load_state_dict(torch.load(os.path.join(model_dir, 'class_modulation.pth')))
		self.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
		self.encoders.init()

		self.model = LordStage2(self.encoders, self.modulation, self.generator)

		self.model.to(self.device)
		self.embeddings.to(self.device)

		criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)

		optimizer = Adam([
			{
				'params': self.model.encoders.parameters(),
				'lr': self.config['train_encoders']['learning_rate']
			},
			{
				'params': self.model.modulation.parameters(),
				'lr': self.config['train_encoders']['learning_rate']
			},
			{
				'params': self.model.generator.parameters(),
				'lr': self.config['train_encoders']['learning_rate']
			}
		], betas=(0.5, 0.999))

		scheduler = ReduceLROnPlateau(
			optimizer, mode='min', factor=0.5, patience=20, verbose=1
		)

		with SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'stage2')) as summary:
			train_loss = AverageMeter()
			for epoch in range(1, self.config['train']['n_epochs'] + 1):
				self.model.train()
				train_loss.reset()

				with tqdm(iterable=data_loader) as pbar:
					for batch in pbar:
						batch_imgs, batch_img_ids, batch_class_ids = (tensor.to(self.device) for tensor in batch)
						batch_content_codes, batch_class_codes = self.embeddings(batch_img_ids, batch_class_ids)

						generated_imgs, predicted_content_codes, predicted_class_codes = self.model(batch_imgs)

						optimizer.zero_grad()

						perc_loss = criterion(generated_imgs, batch_imgs)
						loss_content = F.mse_loss(batch_content_codes, predicted_content_codes)
						loss_class = F.mse_loss(batch_class_codes, predicted_class_codes)
						loss = perc_loss + 10*loss_content + 10*loss_class
						loss.backward()

						optimizer.step()

						train_loss.update(loss.item())
						pbar.set_description_str('epoch #{}'.format(epoch))
						pbar.set_postfix(loss=train_loss.avg)

				torch.save(self.model.encoders.state_dict(), os.path.join(model_dir, 'encoders.pth'))
				torch.save(self.model.generator.state_dict(), os.path.join(model_dir, 'generator.pth'))
				torch.save(self.model.modulation.state_dict(), os.path.join(model_dir, 'class_modulation.pth'))

				scheduler.step(train_loss.avg)

				self.model.eval()
				fixed_sample_img = self.encoder_evaluate(imgs, randomized=False)
				random_sample_img = self.encoder_evaluate(imgs, randomized=True)

				summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)
				summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step=epoch)
				summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step=epoch)

	def evaluate(self, imgs, img_ids, class_ids, n_samples=5, randomized=False):
		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed=1234)

		img_idx = random.choice(imgs.size(0), size=n_samples, replace=False)
		imgs, img_ids, class_ids = (imgs[img_idx], img_ids[img_idx].to(self.device), class_ids[img_idx].to(self.device))

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=2)]
		for i in range(n_samples):
			converted_imgs = [imgs[i]] + [
				self.model(img_ids[[j]], class_ids[[i]])[0][0].detach().cpu()
				for j in range(n_samples)
			]

			output.append(np.concatenate(converted_imgs, axis=2))

		return np.concatenate(output, axis=1)

	def encoder_evaluate(self, imgs, n_samples=5, randomized=False):
		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed=1234)

		img_idx = random.choice(imgs.size(0), size=n_samples, replace=False)
		imgs = imgs[img_idx]

		imgs_cuda = imgs.to(self.device)

		content_code, class_code = self.model.encoders(imgs_cuda)
		class_adain_params = self.model.modulation(class_code)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=2)]
		for i in range(n_samples):
			converted_imgs = [imgs[i]] + [
				self.model.generator(content_code[[j]], class_adain_params[[i]])[0].detach().cpu()
				for j in range(n_samples)
			]

			output.append(np.concatenate(converted_imgs, axis=2))

		return np.concatenate(output, axis=1)
