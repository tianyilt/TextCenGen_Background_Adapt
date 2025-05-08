import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import re
from datetime import datetime

def save_cmd(save_name, input_params, output_dir):
	"""
	:param save_name: str
	:param input_params: dict
	:param output_dir: str
	:return:
	"""
	with open(os.path.join(output_dir, save_name + '.txt'), 'w') as f:
		for k, v in input_params.items():
			f.write(f'{k}: {v}\n')

def prompt2filename(idx, prompt, object_to_edit, seed):
	"""
	replace " " and "," with "_"
	:param prompt:
	:param object_to_edit:
	:param seed:
	:return:
	"""
	prompt = re.sub(r'[^\w\s]', '', prompt)
	# prevent prompt too long
	prompt = prompt[:50]
	object_to_edit = re.sub(r'[^\w\s]', '', object_to_edit)
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
	return str(idx) + '_' + prompt + '_' + object_to_edit + "_" + current_time + '_' + str(seed)

def resave_aux_key(module, *args, old_key="attn", new_key="last_attn"):
	module._aux[new_key] = module._aux[old_key]

_SG_RES = 64

def resize(x):
	return TF.resize(x, _SG_RES, antialias=True)

def stash_to_aux(module, args, kwargs, output, mode, key="last_feats", args_idx=None, kwargs_key=None, fn_to_run=None):
	to_save = None
	if mode == "args":
		to_save = input
		if args_idx is not None:
			to_save = args[args_idx]
	elif mode == "kwargs":
		assert kwargs_key is not None
		to_save = kwargs[kwargs_key]
	elif mode == "output":
		to_save = output
	if fn_to_run is not None:
		to_save = fn_to_run(to_save)
	try:
		global save_aux
		if not save_aux:
			len_ = len(module._aux[key])
			del module._aux[key]
			module._aux[key] = [None] * len_ + [to_save]
		else:
			module._aux[key][-1] = module._aux[key][-1].cpu()
			module._aux[key].append(to_save)
	except:
		try:
			del module._aux[key]
		except:
			pass
		module._aux = {key: [to_save]}

def visualize_attention_maps(attn_scores, output_dir, tokenizer, prompt, num_tokens=20, fig_size=(16, 20), cmap='hot', cur_step=None, place_in_unet=None, upsample_size=(256, 256)):
	"""
	将注意力分数张量可视化为热力图网格,并保存到指定目录。

	Args:
		attn_scores (torch.Tensor): 注意力分数张量,形状为 [1, num_heads, seq_len, seq_len]
		output_dir (str): 保存可视化结果的目录
		tokenizer (PreTrainedTokenizer): 用于将prompt转换为token的tokenizer
		prompt (str): 输入的prompt文本
		num_tokens (int, optional): 要可视化的token数量。默认为20。
		fig_size (tuple, optional): 图像的尺寸。默认为 (16, 20)。
		cmap (str, optional): 热力图的颜色映射。默认为 'hot'。
		cur_step (int, optional): 当前的训练步骤。默认为None。
		place_in_unet (str, optional): 当前注意力层在UNet中的位置。默认为None。
		upsample_size (tuple, optional): 上采样后的图像大小。默认为 (256, 256)。
	"""
	# 如果当前步骤不是50的倍数,则不保存图片
	if cur_step is None or cur_step % 50 != 0:
		return

	# 对第1维(num_heads)取平均
	attn_scores_mean = attn_scores.mean(dim=0)

	# 重塑为 [1, seq_len, seq_len, 1]
	attn_scores_reshaped = attn_scores_mean.reshape(1, attn_scores_mean.size(-2), attn_scores_mean.size(-1), 1)

	# 如果是attn2,则对条件向量进行上采样
	if 'attn2' in place_in_unet:
		attn_scores_reshaped = F.interpolate(attn_scores_reshaped, size=(attn_scores_reshaped.size(-2), upsample_size[1]), mode='bilinear', align_corners=False)

	# 使用双线性插值对注意力分数张量进行上采样
	attn_scores_upsampled = F.interpolate(attn_scores_reshaped, size=upsample_size, mode='bilinear', align_corners=False)

	# 将上采样后的张量重塑为 [seq_len, seq_len]
	attn_scores_upsampled = attn_scores_upsampled.reshape(attn_scores_upsampled.size(1), attn_scores_upsampled.size(2))

	# 创建目录,如果不存在
	os.makedirs(output_dir, exist_ok=True)

	# 计算子图的行列数
	num_rows = (num_tokens - 1) // 4 + 1
	num_cols = min(num_tokens, 4)

	# 创建画布和子图网格
	fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=fig_size)

	# 在图像上方添加一个文本框,显示注意力分数张量的形状和prompt
	fig.suptitle(f"Attention Scores Shape: {attn_scores.shape}\nPrompt: {prompt}", fontsize=16)

	# 调整子图位置,为文本框留出空间
	fig.subplots_adjust(top=0.9)

	# 将所有子图展平为一维数组
	axes = axes.flatten()

	# 将prompt转换为token列表
	prompt_tokens = tokenizer.tokenize(prompt)

	# 遍历指定数量的token
	for token_idx in range(min(num_tokens, len(prompt_tokens))):
		# 选择当前token对应的注意力图
		attn_scores_token = attn_scores_upsampled[:, :, token_idx]

		# 将张量从GPU移动到CPU,并从计算图中分离
		attn_scores_token = attn_scores_token.cpu().detach()

		# 获取当前子图的坐标轴
		ax = axes[token_idx]

		# 绘制热力图
		im = ax.matshow(attn_scores_token, cmap=cmap)

		# 添加颜色条
		cbar = ax.figure.colorbar(im, ax=ax)

		# 设置坐标轴和标题
		ax.axis('off')
		ax.set_title(prompt_tokens[token_idx])

	# 调整子图间距
	plt.tight_layout()

	# 将place_in_unet中的'.'替换为'_',以避免在文件名中出现'.'
	place_in_unet = re.sub(r'\.', '_', place_in_unet)

	# 保存合并后的图片,文件名中包含当前步骤数和place_in_unet信息
	plt.savefig(os.path.join(output_dir, f"attn_map_merged_step_{cur_step}_{place_in_unet}.png"), bbox_inches='tight', pad_inches=0)

	# 关闭画布
	plt.close(fig)    
