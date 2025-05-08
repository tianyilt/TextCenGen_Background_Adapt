import math
import numpy as np
import torch

from .functions import normalize
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging

logger = logging.getLogger("global_logger")
output_dir=''
def threshold_attention(attn, s=10):
	norm_attn = s * (normalize(attn) - 0.5)
	return normalize(norm_attn.sigmoid())

def get_shape(attn, s=20):
	return threshold_attention(attn, s)

def find_missing_numbers(lst, y):
    full_set = set(y)
    lst_set = set(lst)
    missing_numbers = sorted(list(full_set - lst_set))
    return missing_numbers

def getConflictAreas(x, y, a, b, epoch, attn_storage, all_index, height=512, width=512,threshold_low_tvloss_token=0.025):
    ori_attn_down = attn_storage.attention_store['ori']['down']
    ori_attn_up = attn_storage.attention_store['ori']['up']
    ori_attn_mid = attn_storage.attention_store['ori']['mid']
    final_h = int(math.sqrt(max([x.shape[1] for x in ori_attn_down])))
    res_up = []
    other = []
    res_down = []
    x = max(int(x - 0.03 * final_h), 0)
    y = max(int(y - 0.03 * final_h), 0)
    a = min(int(a + 0.03 * final_h), final_h - 1)
    b = min(int(b + 0.03 * final_h), final_h - 1)
    target_h, target_w = 512, 384  # 推荐的大小
    # show_attention_map(x, y, a, b, ori_attn_up, last_index, i)
    for idx in all_index:
        attn = []
        for ori_attn_map in ori_attn_down:
            ori_attn_map = torch.unsqueeze(torch.mean(ori_attn_map, dim=0), dim=0)
            attn_map = ori_attn_map[:, :, idx]
            num_elements = attn_map.numel()
            scale = height*width//num_elements
            h = height // int(math.sqrt(scale))
            w = width// int(math.sqrt(scale))
            
            if h * w != num_elements:
                raise ValueError(f"Cannot reshape attn_map of numel {num_elements} to a shape where one dimension is a multiple of 64")

            attn_map = attn_map.reshape(h, w)
            attn_map = torch.unsqueeze(torch.unsqueeze(attn_map, dim=0), dim=0)
            if h < final_h:
                attn_map = F.interpolate(attn_map, size=(final_h, final_h), mode='bilinear', align_corners=False)
            attn.append(attn_map)

        attn = [F.interpolate(attn_map, size=(target_h, target_w), mode='bilinear', align_corners=False) for attn_map in attn]
        attn_map = torch.squeeze(torch.mean(torch.cat(attn, dim=0), dim=0))
        attn_map = get_shape(attn_map)
        attn_map_in = attn_map[x:a + 1, y:b + 1]
        # map = attn_map_in.detach().cpu()
        # plt.imshow(map)
        # plt.savefig('test0.png')
        # plt.close()
        attn_map_in=attn_map_in.detach().cpu().numpy()
        pixel_diff_x = np.diff(attn_map_in, axis=0)
        pixel_diff_y = np.diff(attn_map_in, axis=1)
        tv_loss = (np.sum(np.abs(pixel_diff_x)) + np.sum(np.abs(pixel_diff_y)))/attn_map_in.shape[0]/attn_map_in.shape[1]
        attn_map_in[attn_map_in < 0.5] = 0
        mask = attn_map_in > 0
        positive_values = attn_map_in[mask]
        average_positive=0
        if positive_values.size>0:
            average_positive = positive_values.mean()
        positive_area_size = positive_values.size
        # ratio_of_frame = (torch.sum(attn_map_in) / torch.sum(attn_map)).item()
        # if ratio_of_frame >= 0.14:
        #     res.append(idx)
        # Compute the total variation loss
        
        if tv_loss >threshold_low_tvloss_token and ((positive_area_size / (a - x) / (b - y) >= 0.14 if epoch < 20 else False) or average_positive >= 0.8):
            res_up.append(idx)
        else:
            other.append(idx)
    # for idx in all_index:
    #     attn = []
    #     for ori_attn_map in ori_attn_down:
    #         ori_attn_map = torch.unsqueeze(torch.mean(ori_attn_map, dim=0), dim=0)
    #         h = int(math.sqrt(ori_attn_map.shape[-2]))
    #         w = h
    #         attn_map = ori_attn_map[:, :, idx]
    #         attn_map = attn_map.reshape(h, w)
    #         attn_map = torch.unsqueeze(torch.unsqueeze(attn_map, dim=0), dim=0)
    #         if h < final_h:
    #             attn_map = F.interpolate(attn_map, size=(final_h, final_h), mode='bilinear', align_corners=False)
    #         attn.append(attn_map)
    #     attn_map = torch.squeeze(torch.mean(torch.cat(attn, dim=0), dim=0))
    #     attn_map = get_shape(attn_map)
    #     attn_map_in = attn_map[x:a + 1, y:b + 1]
    #     attn_map_in[attn_map_in < 0.5] = 0
    #     mask = attn_map_in > 0
    #     positive_values = attn_map_in[mask]
    #     average_positive = positive_values.mean()
    #     positive_area_size = positive_values.numel()
    #     # ratio_of_frame = (torch.sum(attn_map_in) / torch.sum(attn_map)).item()
    #     # if ratio_of_frame >= 0.14:
    #     #     res.append(idx)
    #     if positive_area_size / (a - x) / (b - y) >= 0.14 or average_positive >= 0.8:
    #         res_down.append(idx)
    # res = sorted(list(set(res_up).intersection(set(res_down))))
    
    # logger.info('res_up:{}'.format(res_up))
    # logger.info('res_down:{}'.format(res_down))
    return torch.tensor(res_up), torch.tensor(other)


def get_output_dir(dir):
    global output_dir
    output_dir = dir

def show_attention_map(attn_storage,i,all_index,x=-1,y=-1,a=-1,b=-1):
    ori_attn_up = attn_storage.attention_store['ori']['up']
    col = 4
    row = math.ceil(len(all_index) / col)
    final_h = int(math.sqrt(max([x.shape[1] for x in ori_attn_up])))
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(row, col, figsize=(20, 20))
    edge_value = 1.0
    last_index=len(all_index)
    for idx in range(row * col):
        if idx < last_index:
            attn = []
            for ori_attn_map in ori_attn_up:
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
            # 在张量上绘制框
            # 水平边
            if x>=0:
                attn_map[x, y:b + 1] = edge_value  # 上边
                attn_map[a, y:b + 1] = edge_value  # 下边
                # 垂直边
                attn_map[x:a + 1, y] = edge_value  # 左边
                attn_map[x:a + 1, b] = edge_value  # 右边
            attn_map = attn_map.detach().cpu()
        else:
            attn_map = np.zeros((h, w), dtype=np.uint8)

        axs[int(idx/ col)][idx % col].imshow(attn_map, cmap='jet')
        # 隐藏X轴刻度
        axs[int(idx / col)][idx % col].set_xticks([])
        # 隐藏Y轴刻度
        axs[int(idx / col)][idx % col].set_yticks([])
    plt.savefig(output_dir + '/epoch_{}.png'.format(i))
    plt.show()

