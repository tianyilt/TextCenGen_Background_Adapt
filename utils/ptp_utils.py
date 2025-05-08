from diffusers.models.attention_processor import AttnProcessor, Attention
import torch
from functools import partial
import fastcore.all as fc
import numpy as np
from typing import List
from PIL import Image
from .vis_utils import preprocess_image
import abc
from typing import Union, Tuple, List, Dict, Optional
import torch.nn.functional as nnf


def get_features(hook, layer, inp, out):
	if not hasattr(hook, 'feats'): hook.feats = out
	hook.feats = out

def fill_tensor(left, top, right, bottom,width, height,batch_size=1):
    tensor = torch.zeros((batch_size,height, width,1), dtype=torch.float16)
    tensor[:,top:bottom+1, left:right+1,0] = 1
    return tensor

def get_latents_from_image(pipe, img_path, device):
	if img_path is None: return None
	img = Image.open(img_path)
	img = img.convert('RGB')
	image = preprocess_image(img).to(device)
	init_latents = pipe.vae.encode(image.half()).latent_dist.sample() * 0.18215
	shape = init_latents.shape
	noise = torch.randn(shape, device=device)
	timesteps = pipe.scheduler.timesteps[0]
	timesteps = torch.tensor([timesteps], device=device)
	init_latents = pipe.scheduler.add_noise(init_latents, noise, timesteps).half()
	return init_latents


class Hook():
	def __init__(self, model, func): self.hook = model.register_forward_hook(partial(func, self))

	def remove(self): self.hook.remove()

	def __del__(self): self.remove()


class AttentionStore:
	@staticmethod
	def get_empty_store():
		return {'ori': {"down": [], "mid": [], "up": []}, 'edit': {"down": [], "mid": [], "up": []}}

	def __init__(self, attn_res=[4096, 1024, 256, 64]):
		"""
		Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
		process
		"""
		self.num_att_layers = -1
		self.cur_att_layer = 0
		self.step_store = self.get_empty_store()
		self.attention_store = {}
		self.curr_step_index = 0
		self.attn_res = attn_res
		self.cur_step = 0

	def __call__(self, attention_map, is_cross, place_in_unet: str, pred_type='ori'):
		# if not name in self.step_store:
		#     self.step_store[name] = {}
		# self.step_store[name][pred_type] = attention_map
		if self.cur_att_layer >= 0 and is_cross:
			if attention_map.shape[1] in self.attn_res:
				self.step_store[pred_type][place_in_unet].append(attention_map)
		self.cur_att_layer += 1
		if self.cur_att_layer == self.num_att_layers:
			self.cur_att_layer = 0
			self.cur_step += 1
			self.between_steps(pred_type)

	def aggregate_attention(self, from_where: List[str], pred_type='ori', res=64) -> torch.Tensor:
		"""Aggregates the attention across the different layers and heads at the specified resolution."""
		out = []
		# attention_maps = self.get_average_attention()
		attention_maps = self.attention_store[pred_type]
		for location in from_where:
			for item in attention_maps[location]:
				cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
				out.append(cross_maps)
		out = torch.cat(out, dim=0)
		out = out.sum(0) / out.shape[0]
		return out

	def get_average_attention(self):
		average_attention = {key: [item for item in self.attention_store[key]] for key in
							 self.attention_store}
		return average_attention

	def reset(self):
		self.cur_att_layer = 0
		self.step_store = self.get_empty_store()
		self.attention_store = {}

	def maps(self, block_type: str):
		return self.attention_store[block_type]

	def between_steps(self, pred_type='ori'):
		self.attention_store[pred_type] = self.step_store[pred_type]
		self.step_store = self.get_empty_store()


class CustomAttnProcessor(AttnProcessor):
	def __init__(self, attnstore, place_in_unet=None):
		super().__init__()
		fc.store_attr()
		self.attnstore = attnstore
		self.place_in_unet = place_in_unet
		self.store = False

	def set_storage(self, store, pred_type):
		self.store = store
		self.pred_type = pred_type

	def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)
		attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
		query = attn.to_q(hidden_states)

		is_cross = encoder_hidden_states is not None
		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)

		if self.store:
			self.attnstore(attention_probs, is_cross, self.place_in_unet,
						   pred_type=self.pred_type)  ## stores the attention maps in attn_storage

		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		return hidden_states


class LocalBlend:

	def __call__(self, x_t, attention_store):
		k = 1
		maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
		maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.max_num_words) for item in maps]
		maps = torch.cat(maps, dim=1)
		maps = (maps * self.alpha_layers).sum(-1).mean(1)
		mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
		mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
		mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
		mask = mask.gt(self.threshold)
		mask = (mask[:1] + mask[1:]).float()
		x_t = x_t[:1] + mask * (x_t - x_t[:1])
		return x_t

	def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, device, threshold=.3, max_num_words=77):
		self.max_num_words = 77

		alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
		for i, (prompt, words_) in enumerate(zip(prompts, words)):
			if type(words_) is str:
				words_ = [words_]
			for word in words_:
				ind = get_word_inds(prompt, word, tokenizer)
				alpha_layers[i, :, :, :, :, ind] = 1
		self.alpha_layers = alpha_layers.to(device)
		self.threshold = threshold


class AttentionControlEdit(AttentionStore, abc.ABC):

	def step_callback(self, x_t):
		if self.local_blend is not None:
			x_t = self.local_blend(x_t, self.attention_store)
		return x_t

	def replace_self_attention(self, attn_base, att_replace):
		if att_replace.shape[2] <= 16 ** 2:
			return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
		else:
			return att_replace

	@abc.abstractmethod
	def replace_cross_attention(self, attn_base, att_replace):
		raise NotImplementedError

	def forward(self, attn, is_cross: bool, place_in_unet: str):
		super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
		# FIXME not replace correctly
		if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
			h = attn.shape[0] // (self.batch_size)
			attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
			attn_base, attn_repalce = attn[0], attn[1:]
			if is_cross:
				alpha_words = self.cross_replace_alpha[self.cur_step]
				attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
					1 - alpha_words) * attn_repalce
				attn[1:] = attn_repalce_new
			else:
				attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
			attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
		return attn

	def __init__(self, prompts, num_steps: int,
				 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
				 self_replace_steps: Union[float, Tuple[float, float]],
				 local_blend: Optional[LocalBlend],
				 tokenizer,
				 device):
		super(AttentionControlEdit, self).__init__()
		# add tokenizer and device here

		self.tokenizer = tokenizer
		self.device = device

		self.batch_size = len(prompts)
		self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
																  self.tokenizer).to(self.device)
		if type(self_replace_steps) is float:
			self_replace_steps = 0, self_replace_steps
		self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
		self.local_blend = local_blend  # 在外面定义后传进来


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
	words_x = x.split(' ')
	words_y = y.split(' ')
	if len(words_x) != len(words_y):
		raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
						 f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
	inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
	inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
	inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
	mapper = np.zeros((max_len, max_len))
	i = j = 0
	cur_inds = 0
	while i < max_len and j < max_len:
		if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
			inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
			if len(inds_source_) == len(inds_target_):
				mapper[inds_source_, inds_target_] = 1
			else:
				ratio = 1 / len(inds_target_)
				for i_t in inds_target_:
					mapper[inds_source_, i_t] = ratio
			cur_inds += 1
			i += len(inds_source_)
			j += len(inds_target_)
		elif cur_inds < len(inds_source):
			mapper[i, j] = 1
			i += 1
			j += 1
		else:
			mapper[j, j] = 1
			i += 1
			j += 1

	return torch.from_numpy(mapper).float()


def get_word_inds(text: str, word_place: int, tokenizer):
	split_text = text.split(" ")
	if type(word_place) is str:
		word_place = [i for i, word in enumerate(split_text) if word_place == word]
	elif type(word_place) is int:
		word_place = [word_place]
	out = []
	if len(word_place) > 0:
		words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
		cur_len, ptr = 0, 0

		for i in range(len(words_encode)):
			cur_len += len(words_encode[i])
			if ptr in word_place:
				out.append(i + 1)
			if cur_len >= len(split_text[ptr]):
				ptr += 1
				cur_len = 0
	return np.array(out)


def get_time_words_attention_alpha(prompts, num_steps,
								   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
								   tokenizer, max_num_words=77):
	if type(cross_replace_steps) is not dict:
		cross_replace_steps = {"default_": cross_replace_steps}
	if "default_" not in cross_replace_steps:
		cross_replace_steps["default_"] = (0., 1.)
	alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
	for i in range(len(prompts) - 1):
		alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
												  i)
	for key, item in cross_replace_steps.items():
		if key != "default_":
			inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
			for i, ind in enumerate(inds):
				if len(ind) > 0:
					alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
	alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
	return alpha_time_words


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
						   word_inds: Optional[torch.Tensor] = None):
	if type(bounds) is float:
		bounds = 0, bounds
	start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
	if word_inds is None:
		word_inds = torch.arange(alpha.shape[2])
	alpha[: start, prompt_ind, word_inds] = 0
	alpha[start: end, prompt_ind, word_inds] = 1
	alpha[end:, prompt_ind, word_inds] = 0
	return alpha


class AttentionReplace(AttentionControlEdit):

	def replace_cross_attention(self, attn_base, att_replace):
		return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

	def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
				 local_blend: Optional[LocalBlend] = None, tokenizer=None, device=None):
		super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,
											   tokenizer, device)
		self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
	words_x = x.split(' ')
	words_y = y.split(' ')
	if len(words_x) != len(words_y):
		raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
						 f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
	inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
	inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
	inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
	mapper = np.zeros((max_len, max_len))
	i = j = 0
	cur_inds = 0
	while i < max_len and j < max_len:
		if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
			inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
			if len(inds_source_) == len(inds_target_):
				mapper[inds_source_, inds_target_] = 1
			else:
				ratio = 1 / len(inds_target_)
				for i_t in inds_target_:
					mapper[inds_source_, i_t] = ratio
			cur_inds += 1
			i += len(inds_source_)
			j += len(inds_target_)
		elif cur_inds < len(inds_source):
			mapper[i, j] = 1
			i += 1
			j += 1
		else:
			mapper[j, j] = 1
			i += 1
			j += 1

	return torch.from_numpy(mapper).float()


def get_replacement_mapper(prompts, tokenizer, max_len=77):
	x_seq = prompts[0]
	mappers = []
	for i in range(1, len(prompts)):
		mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
		mappers.append(mapper)
	return torch.stack(mappers)
