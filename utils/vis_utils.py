import fastcore.all as fc
from PIL import Image
import math, random, torch, matplotlib.pyplot as plt, numpy as np, matplotlib as mpl
from itertools import zip_longest
from diffusers.utils import PIL_INTERPOLATION
from math import sqrt
import os
from sklearn.decomposition import PCA
import torchvision.transforms as T
import json
from torch import tensor
from datetime import datetime
import torch.nn.functional as F
from utils.functions import normalize
from PIL import Image, ImageDraw

def check_and_detach(tensor: torch.Tensor):
	if tensor.requires_grad:
		return tensor.detach()
	else:
		return tensor
@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, save_orig=False, **kwargs):
	"Show a PIL or PyTorch image on `ax`."
	if save_orig: im.save('orig.png')
	if fc.hasattrs(im, ('cpu', 'permute', 'detach')):
		im = im.detach().cpu()
		if len(im.shape) == 3 and im.shape[0] < 5: im = im.permute(1, 2, 0)
	elif not isinstance(im, np.ndarray):
		im = np.array(im)
	if im.shape[-1] == 1: im = im[..., 0]
	if ax is None: _, ax = plt.subplots(figsize=figsize)
	ax.imshow(im, **kwargs)
	if title is not None: ax.set_title(title)
	ax.set_xticks([])
	ax.set_yticks([])
	if noframe: ax.axis('off')
	return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
	nrows: int = 1,  # Number of rows in returned axes grid
	ncols: int = 1,  # Number of columns in returned axes grid
	figsize: tuple = None,  # Width, height in inches of the returned figure
	imsize: int = 3,  # Size (in inches) of images that will be displayed in the returned figure
	suptitle: str = None,  # Title to be set to returned figure
	**kwargs
):  # fig and axs
	"A figure and set of subplots to display images of `imsize` inches"
	if figsize is None: figsize = (ncols * imsize, nrows * imsize)
	fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
	if suptitle is not None: fig.suptitle(suptitle)
	if nrows * ncols == 1: ax = np.array([ax])
	return fig, ax


@fc.delegates(subplots)
def get_grid(
	n: int,  # Number of axes
	nrows: int = None,  # Number of rows, defaulting to `int(math.sqrt(n))`
	ncols: int = None,  # Number of columns, defaulting to `ceil(n/rows)`
	title: str = None,  # If passed, title set to the figure
	weight: str = 'bold',  # Title font weight
	size: int = 14,  # Title font size
	**kwargs,
):  # fig and axs
	"Return a grid of `n` axes, `rows` by `cols`"
	if nrows:
		ncols = ncols or int(np.floor(n / nrows))
	elif ncols:
		nrows = nrows or int(np.ceil(n / ncols))
	else:
		nrows = int(math.sqrt(n))
		ncols = int(np.floor(n / nrows))
	fig, axs = subplots(nrows, ncols, **kwargs)
	for i in range(n, nrows * ncols): axs.flat[i].set_axis_off()
	if title is not None: fig.suptitle(title, weight=weight, size=size)
	return fig, axs


@fc.delegates(subplots)
def show_images(ims: list,  # Images to show
				nrows: int = None,  # Number of rows in grid
				ncols: int = None,  # Number of columns in grid (auto-calculated if None)
				titles: list = None,  # Optional list of titles for each image
				save_orig: bool = False,  # If True, save original image
				**kwargs):
	"Show all images `ims` as subplots with `rows` using `titles`"
	axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
	for im, t, ax in zip_longest(ims, titles or [], axs): show_image(im, ax=ax, title=t, save_orig=save_orig)


def save_images(ims, folder, input_params, nrows=None, ncols=None, titles=None, save_combined=True,left_top=None,right_bottom=None,save_with_draw_frame=False,**kwargs):
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Determine the size of the grid
	nrows = nrows or int(math.sqrt(len(ims)))
	ncols = ncols or int(math.ceil(len(ims) / nrows))

	# Save individual images, first one named 'edited.png', second one named 'origin.png'
	for idx, im in enumerate(ims):
		if idx == 0:
			im_path = os.path.join(folder, titles[0]+'.png')
		elif idx == 1:
			im_path = os.path.join(folder, titles[1]+'.png')
		else:
			im_path = os.path.join(folder, f'image_{idx}.png')
		im.save(im_path)

	# Create a combined image
	if save_combined:
		combined_width = ncols * ims[0].width
		combined_height = nrows * ims[0].height
		combined_image = Image.new('RGB', (combined_width, combined_height))

		for idx, im in enumerate(ims):
			x = (idx % ncols) * im.width
			y = (idx // ncols) * im.height
			if idx==1:
				draw = ImageDraw.Draw(im)
				draw.rectangle([left_top, right_bottom], outline="red", width=1)
			combined_image.paste(im, (x, y))
		current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
		combined_image_path = os.path.join(folder, 'dec_{}combined_image.png'.format(current_time))
		combined_image.save(combined_image_path)
	if save_with_draw_frame:
		combined_image = Image.new('RGB', (ims[0].width, ims[0].width))
		draw = ImageDraw.Draw(ims[0])
		draw.rectangle([left_top, right_bottom], outline="red", width=1)
		combined_image.paste(ims[0], (0,0))
		current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
		combined_image_path = os.path.join(folder, 'image_with_frame_{}.png'.format(current_time))
		combined_image.save(combined_image_path)
	# Save input parameters into a JSON file
	params_path = os.path.join(folder, 'input_parameters.json')
	with open(params_path, 'w') as json_file:
		json.dump(input_params, json_file, indent=4)


def preprocess_image(image):
	w, h = image.size
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.0 * image - 1.0


def visualize_and_save_features_pca(feature_maps_fit_data, feature_maps_transform_data, transform_experiments, t):
	feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
	pca = PCA(n_components=3)
	pca.fit(feature_maps_fit_data)
	feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
	feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
	for i, experiment in enumerate(transform_experiments):
		pca_img = feature_maps_pca[i]  # (H * W) x 3
		h = w = int(sqrt(pca_img.shape[0]))
		pca_img = pca_img.reshape(h, w, 3)
		pca_img_min = pca_img.min(axis=(0, 1))
		pca_img_max = pca_img.max(axis=(0, 1))
		pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
		pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
		pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
		pca_img.save(os.path.join(f"{experiment}_time_{t}.png"))


def save_different_attantion_map(atten_tensor, output_dir: str, name: str = "i_1_layer_10", is_show: bool = False,
								 row=1, col=0):
	h = int(torch.sqrt(torch.tensor(max([tensor.shape[-2] for tensor in atten_tensor]), dtype=torch.float)).item())
	if (col == 0):
		col = math.ceil(len(atten_tensor) / row)
	else:
		row = math.ceil(len(atten_tensor) / col)
	tmp = []
	for x in atten_tensor:
		ori_h = int(sqrt(x.shape[1]))
		y = x.reshape(1, 1, ori_h, ori_h)
		tmp.append(F.interpolate(y, size=(h, h), mode='bilinear', align_corners=False))
	tmp = torch.cat(tmp, dim=0)
	tmp = tmp.detach().cpu()

	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(row, col, figsize=(20, 20))
	if (len(atten_tensor) == 1):
		plt.imshow(tmp[0][0], cmap='jet')
	elif row == 1 or col == 1:
		for i in range(row if col == 1 else col):
			# 获取第i个图像
			img = tmp[i, 0, :, :] if i < len(atten_tensor) else np.zeros((h, h, 1), dtype=np.uint8)
			# 使用matplotlib来绘制图像
			axs[i].imshow(img, cmap='jet')
	else:
		for i in range(row):
			for j in range(col):
				# 获取第i个图像
				img = tmp[i * col + j, 0, :, :] if i * col + j < len(atten_tensor) else np.zeros((h, h, 1),
																								 dtype=np.uint8)
				# 使用matplotlib来绘制图像
				axs[i][j].imshow(img, cmap='jet')
	# save
	plt.savefig(os.path.join(output_dir, f"{name}.png"))
	plt.close()


def show_different_attantion_map(ori_attn_store, row=1, col=0):
	attn_store = []
	for _, v in ori_attn_store.attention_store['ori'].items():
		attn_store.extend(v)
	h = int(torch.sqrt(torch.tensor(max([tensor.shape[-2] for tensor in attn_store]), dtype=torch.float)).item())
	if (col == 0):
		col = math.ceil(len(attn_store) / row)
	else:
		row = math.ceil(len(attn_store) / col)
	tmp = []
	for x in attn_store:
		x = torch.unsqueeze(torch.mean(get_shape(x), dim=0), dim=0)
		ori_h = int(sqrt(x.shape[1]))
		y = x.reshape(1, 1, ori_h, ori_h)
		tmp.append(F.interpolate(y, size=(h, h), mode='bilinear', align_corners=False))
	tmp = torch.cat(tmp, dim=0)
	tmp = tmp.detach().cpu()
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(row, col, figsize=(20, 20))
	if (len(attn_store) == 1):
		axs.imshow(tmp[0][0], cmap='jet')
	elif row == 1 or col == 1:
		for i in range(row if col == 1 else col):
			img = tmp[i, 0, :, :]
			axs[i].imshow(img, cmap='jet')
	else:
		for i in range(row):
			for j in range(col):

				img = tmp[i * col + j, 0, :, :] if i * col + j < len(attn_store) else np.zeros((h, h, 1),
																							   dtype=np.uint8)

				axs[i][j].imshow(img, cmap='jet')
	# save
	plt.show()


def save_attantion_map(atten_tensor: torch.Tensor, output_dir: str, name: str = "i_1_layer_10", is_show: bool = False,
					   row=1, col=0):
	atten_tensor = check_and_detach_tensor(atten_tensor)
	h = w = int(torch.sqrt(torch.tensor([atten_tensor.shape[-2]], dtype=torch.float)).item())
	tmp = atten_tensor.reshape(atten_tensor.shape[0], 1, h, h)
	import matplotlib.pyplot as plt
	if (col == 0):
		col = int(atten_tensor.shape[0] / row)
	else:
		row = int(atten_tensor.shape[0] / col)
	fig, axs = plt.subplots(row, col, figsize=(20, 20))
	if (atten_tensor.shape[0] == 1):
		axs.imshow(tmp[0][0], cmap='jet')
	elif row == 1 or col == 1:
		for i in range(row if col == 1 else col):
			# 获取第i个图像
			img = tmp[i, 0, :, :]
			# 使用matplotlib来绘制图像
			axs[i].imshow(img, cmap='jet')
	else:
		for i in range(row):
			for j in range(col):
				# 获取第i个图像
				img = tmp[i, 0, :, :]
				# 使用matplotlib来绘制图像
				axs[i][j].imshow(img, cmap='gray')
	# save
	plt.savefig(os.path.join(output_dir, f"{name}.png"))
	plt.close()


def visualize_attantion_map(atten_tensor: torch.Tensor, output_dir=None, name: str = "i_1_layer_10",
							is_show: bool = False):
	"""
	visualize attention map of one layer of one token
	case:
		visualize_attantion_map(get_shape(orig))
		visualize_attantion_map(move_shape(get_shape(orig)))
		visualize_attantion_map(get_shape(edit))
	:param atten_tensor: shape like (8, h*w, 1)
	:return:
	"""
	atten_tensor = check_and_detach_tensor(atten_tensor)
	h = w = int(torch.sqrt(torch.tensor([atten_tensor.shape[-2]], dtype=torch.float)).item())
	tmp = atten_tensor.reshape(atten_tensor.shape[0], 1, h, h)
	import matplotlib.pyplot as plt

	fig, axs = plt.subplots(1, atten_tensor.shape[0], figsize=(20, 20))

	for i in range(atten_tensor.shape[0]):
		# 获取第i个图像
		img = tmp[i, 0, :, :]

		# 使用matplotlib来绘制图像
		axs[i].imshow(img, cmap='gray')
	# save
	if output_dir is not None:
		plt.savefig(os.path.join(output_dir, f"atten_{name}.png"))
	if is_show:
		plt.show()


def threshold_attention(attn, s=10):
	norm_attn = s * (normalize(attn) - 0.5)
	return normalize(norm_attn.sigmoid())


def get_shape(attn, s=20):
	return threshold_attention(attn, s)


'''
def visualize_attantion_map(atten_tensor: torch.Tensor):
	"""
	visualize attention map of one layer of one token
	case:
		visualize_attantion_map(get_shape(orig))
		visualize_attantion_map(move_shape(get_shape(orig)))
		visualize_attantion_map(get_shape(edit))
	:param atten_tensor: shape like (8, h*w, 1)
	:return:
	"""
	atten_tensor = atten_tensor.detach().cpu()
	h = w = int(torch.sqrt(torch.tensor([atten_tensor.shape[-2]], dtype=torch.float)).item())
	tmp = atten_tensor.reshape(atten_tensor.shape[0], 1, h, h)
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(1, atten_tensor.shape[0], figsize=(20, 20))
	for i in range(atten_tensor.shape[0]):
		# 获取第i个图像
		img = tmp[i, 0, :, :]

		# 使用matplotlib来绘制图像
		axs[i].imshow(img, cmap='gray')
		ax = plt.gca()
		ax.set_xticks([])
		ax.set_yticks([])
	#save
	plt.show()
'''


def visualize_attention_all_layer_all_token(attention_store: dict, output_dir: str, name: str = "epoch_10",
											max_indics: int = 10, is_show_laywise: bool = False):
	"""
	visualize attention map of each layer of all token
	:param attention_store:  attention_store["ori"] = {'down': [Tensor, Tensor, ...], 'up': [Tensor, Tensor, ...], 'down_up': [Tensor, Tensor, ...]}
	each Tensor shape like (shape[0], h*w, 1)
	:param output_dir:
	:param name: save name
	:param max_indics: max indics of token number, less than 77
	:return:
	"""
	import numpy as np

	tokens_per_row = 10

	for k, v in attention_store.items():
		for id, atten_tensor in enumerate(v):
			atten_tensor = check_and_detach_tensor(atten_tensor)
			h = w = int(torch.sqrt(torch.tensor([atten_tensor.shape[-2]], dtype=torch.float)).item())

			rows = int(math.ceil(max_indics / tokens_per_row))
			# Create a new image for every row
			output_image = Image.new('L', (h * tokens_per_row, w * rows))

			for token_id in range(max_indics):
				atten_tensor_token = atten_tensor[:, :, token_id]
				tmp = atten_tensor_token.reshape(atten_tensor.shape[0], 1, h, h)  # (layer_num, 1, h, w)
				if is_show_laywise:
					visualize_attantion_map(atten_tensor_token.unsqueeze(2), output_dir,
											name=f"{name}_layer_{k}_{id}_token_{token_id}", is_show=False)
				avg_img, max_img = get_avg_img_for_each_layer(tmp)  # Tensor: (h,w)

				# Convert avg_img from tensor to PIL Image
				avg_img_pil = Image.fromarray((avg_img.numpy() * 255).astype(np.uint8))

				# Paste avg_img into the output image
				output_image.paste(avg_img_pil, ((token_id % tokens_per_row) * w, (token_id // tokens_per_row) * h))

			# Save the output image
			output_image.save(os.path.join(output_dir, f"{name}_layer_{k}_{id}_tokens_all.png"))


def check_and_detach_tensor(tensor):
	"""
	if cuda tensor, detach it
	:param tensor:
	:return:
	"""
	if tensor.is_cuda:
		tensor = tensor.detach().cpu()
	return tensor


def get_avg_img_for_each_layer(tmp: torch.Tensor, is_show: bool = False):
	"""
	tmp: layer_num, 1, h, w
	:param tmp:
	:return: tensor(h,w) gray img
	"""
	# 计算所有图像的平均值和最大值
	avg_img = torch.mean(tmp, dim=0)[0]
	max_img = torch.max(tmp, dim=0)[0][0]
	if is_show:
		# 在新的子图中显示平均图像和最大值图像
		fig, axs = plt.subplots(1, 2, figsize=(10, 5))
		axs[0].imshow(avg_img, cmap='gray')
		axs[0].set_title('Average Image')
		axs[1].imshow(max_img, cmap='gray')
		axs[1].set_title('Max Image')
		plt.show()
	return avg_img, max_img


def visualize_ori_attention_up_avg_layer_all_token(attn_storage, output_dir, i, all_index, x=-1, y=-1, a=-1, b=-1,
												   is_show=False):
	"""
	:param attn_storage:
	:param output_dir:
	:param i:  epoch
	:param all_index:
	:param x:
	:param y:
	:param a:
	:param b:
	:return:
	"""
	for ori_or_edit in ['ori', 'edit']:
		for location in ['up', 'down', 'mid']:
			attention_map = attn_storage.attention_store[ori_or_edit][location]
			if attention_map == []:
				continue
			output_path = os.path.join(output_dir, f"{ori_or_edit}_{location}_epoch_{i}.png")
			show_attention_map(attention_map, output_path, all_index, x, y, a, b, is_show=is_show)


def show_attention_map(attention_map, output_path, all_index, x=-1, y=-1, a=-1, b=-1, is_show=False, **kwargs):
	"""
	:param attention_map:  list of tensor shape like [num_head, h*w, token_num]
	:param output_path:
	:param all_index:
	:param x:
	:param y:
	:param a:
	:param b:
	:return:
	"""
	col = 4
	row = math.ceil(len(all_index) / col)
	final_h = 64
	plt.clf()
	plt.close()
	fig, axs = plt.subplots(row, col, figsize=(20, 20))
	edge_value = 1.0
	# use last half of tensor for each part, sync to loss in guidance_function
	if attention_map[0].shape[0] == 16:
		is_16 = True
	else:
		is_16 = False
	last_index = len(all_index)
	for idx in range(row * col):
		if idx < last_index:
			attn = []
			for ori_attn_map in attention_map:
				if is_16:
					ori_attn_map = ori_attn_map.chunk(2)[1]
				ori_attn_map = torch.unsqueeze(torch.mean(ori_attn_map, dim=0), dim=0)
				h = int(math.sqrt(ori_attn_map.shape[-2]))
				w = h
				attn_map = ori_attn_map[:, :, all_index[idx]]
				attn_map = attn_map.reshape(h, w)
				attn_map = torch.unsqueeze(torch.unsqueeze(attn_map, dim=0), dim=0)
				if h < final_h:
					attn_map = F.interpolate(attn_map, size=(final_h, final_h), mode='bilinear', align_corners=False)
				attn.append(attn_map)
			attn_map = torch.squeeze(torch.mean(torch.cat(attn, dim=0), dim=0))
			attn_map = get_shape(attn_map)
			# if x, y,a,b is not -1, draw a frame
			if x >= 0:
				# 在张量上绘制框
				if x >= 0:
					# 水平边
					attn_map[x, y:b + 1] = edge_value  # 上边
					attn_map[a, y:b + 1] = edge_value  # 下边
					# 垂直边
					attn_map[x:a + 1, y] = edge_value  # 左边
					attn_map[x:a + 1, b] = edge_value  # 右边
			attn_map = attn_map.detach().cpu()
		else:
			attn_map = np.zeros((h, w), dtype=np.uint8)

		axs[int(idx / col)][idx % col].imshow(attn_map, cmap='jet')
		# 隐藏X轴刻度
		axs[int(idx / col)][idx % col].set_xticks([])
		# 隐藏Y轴刻度
		axs[int(idx / col)][idx % col].set_yticks([])
	plt.savefig(output_path)
	if is_show:
		plt.show()

def sort_filepaths_by_prefix_then_epoch_ascending(filepaths):
	def extract_details(filename):
		parts = filename.split('_')
		prefix = '_'.join(parts[:-2])  # Extracting the prefix
		try:
			epoch = int(parts[-1].split('.')[0])  # Extracting the epoch number
		except ValueError:
			epoch = 0  # Default to 0 if not a number
		return prefix, epoch  # Epoch in ascending order

	# Sorting the filepaths by prefix, then by epoch number in ascending order
	sorted_filepaths = sorted(filepaths, key=lambda x: extract_details(x))
	return sorted_filepaths



def create_montage_with_labels(filepaths, output_filename, image_directory):
	"""

	:param filepaths: ['edit_down_epoch_0.png', 'edit_down_epoch_10.png', 'edit_down_epoch_20.png', 'edit_down_epoch_30.png', 'edit_down_epoch_40.png', 'moved_orig_down_epoch_0.png', 'moved_orig_down_epoch_10.png', 'moved_orig_down_epoch_20.png', 'moved_orig_down_epoch_30.png', 'moved_orig_down_epoch_40.png', 'ori_down_epoch_0.png', 'ori_down_epoch_10.png', 'ori_down_epoch_20.png', 'ori_down_epoch_30.png', 'ori_down_epoch_40.png']
	:param output_filename:
	:param image_directory:
	:return:
	"""
	# Prepare for montage creation
	images_per_epoch = 3  # Number of images per epoch
	epochs = len(filepaths) // images_per_epoch

	# Load images and prepare for labeling
	images = [Image.open(os.path.join(image_directory, fp)) for fp in filepaths if fp is not None]
	widths, heights = zip(*(i.size for i in images))

	# Calculate dimensions for the montage
	max_width = max(widths) * images_per_epoch
	label_height = 30  # Height allocated for labels
	total_height = (max(heights) + label_height) * epochs

	# Prepare font for labels
	try:
		font = ImageFont.truetype("arial.ttf", 20)  # Larger font for better visibility
	except IOError:
		font = ImageFont.load_default()

	# Create a new image for the montage
	montage = Image.new('RGB', (max_width, total_height), (255, 255, 255))
	draw = ImageDraw.Draw(montage)

	for i in range(len(images)):
		prefix_index = i % epochs
		epoch_index = i // epochs

		img = images[i]
		x_offset = epoch_index * img.width
		y_offset = prefix_index * (img.height + label_height)

		montage.paste(img, (x_offset, y_offset))
		label = filepaths[i]
		draw.text((x_offset + 5, y_offset + img.height + 5), label, font=font, fill=(0, 0, 0))

	# Save the montage image
	montage.save(os.path.join(image_directory, output_filename))


def process_files(keyword, output_filename, files):
	files = sorted([f for f in files if keyword in f], key=extract_epoch,
				   reverse=True)
	files = sort_filepaths_by_prefix_then_epoch_ascending(files)
	create_montage_with_labels(files, output_filename, image_directory)


def extract_epoch(filename):
	parts = filename.split('_')
	for part in parts:
		if part.isdigit():
			return int(part)
	return 0


if __name__ == '__main__':
	import os
	from PIL import Image, ImageDraw, ImageFont

	image_directory = "output/TextCenGuidance_ltydev/A lighthouse shines through the thick fog._lighthouse_20240110_170629"
	# Define a helper function to create montage of images with labels
	# List all files in the directory
	files = os.listdir(image_directory)

	process_files("_down_epoch_", 'down_montage_with_labels.png', files)
	process_files("_mid_epoch_", 'mid_montage_with_labels.png', files)
	process_files("_up_epoch_", 'up_montage_with_labels.png', files)
