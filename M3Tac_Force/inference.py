from matplotlib import pyplot as plt
import torch
from data.data import ForceData
import logging
import os
import numpy as np
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from model import get_network
from main import split_dataset
from main import sum_mean_acc
from main import guass_map_cal_delta


def log_creater(log_file_dir):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # set two handlers
    fileHandler = logging.FileHandler(os.path.join(log_file_dir, 'log.log'), mode='w')
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger


def vis_forcemap(rgb, inf, targets, mask, force_maps):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(3, 3, 1)
    ax.imshow(rgb.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    ax.set_title("rgb")

    ax = fig.add_subplot(3, 3, 2)
    ax.imshow(inf.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    ax.set_title("inf")

    ax = fig.add_subplot(3, 3, 3)
    ax.imshow(mask.cpu().detach().numpy().transpose(1, 2, 0) * 255)
    ax.set_title("mask")

    ax = fig.add_subplot(3, 3, 4)
    ax.imshow(targets[0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("x_origin")

    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(targets[1].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("y_origin")

    ax = fig.add_subplot(3, 3, 6)
    ax.imshow(targets[2].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("z_origin")

    ax = fig.add_subplot(3, 3, 7)
    ax.imshow(force_maps[0].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0,
              vmax=255)
    ax.set_title("x_predict")

    ax = fig.add_subplot(3, 3, 8)
    ax.imshow(force_maps[1].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0,
              vmax=255)
    ax.set_title("y_predict")

    ax = fig.add_subplot(3, 3, 9)
    ax.imshow(force_maps[2].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("z_predict")

    fig.savefig("result480.png")


if __name__ == "__main__":

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(42)
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_path = "path_to_your_dataset"
    checkout_path = r"path_to_your_checkpoint"
    checkpoint = torch.load(checkout_path)

    logger = log_creater("logs_path")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_name = 'FSwin_MAP'
    net = get_network(network_name=net_name, logger=logger)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)
    batch_size = 48
    num_workers = 0
    train_dataset_per = 1
    val_dataset_per = 1

    dataset = ForceData(data_path)

    train_dataloader, val_dataloader = split_dataset(dataset, 0.8, batch_size, num_workers, train_dataset_per, val_dataset_per)

    error_force_max = {'error_x': [], 'error_y': [], 'error_z': []}
    error_force_map_max = {'error_x': [], 'error_y': [], 'error_z': []}
    error_delta = {'mask_pix': [], 'a1_x': [], 'a2_x': [], 'a3_x': [], 'a1_y': [], 'a2_y': [], 'a3_y': [], 'a1_z': [], 'a2_z': [], 'a3_z': []}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader), desc='valling'):
            rgb, inf, mask, (x_map_finite, y_map_finite, z_map_finite), (x_map_max, y_map_max, z_map_max), force_max = batch
            inf = inf.to(device)
            mask = mask.to(device)
            x_map_finite, y_map_finite, z_map_finite = x_map_finite.to(device), y_map_finite.to(device), z_map_finite.to(device)
            x_map_max, y_map_max, z_map_max = x_map_max.to(device), y_map_max.to(device), z_map_max.to(device)
            force_max = force_max.to(device)
            output_finite, output_max = net(inf)
            a1_x, a2_x, a3_x, a1_y, a2_y, a3_y, a1_z, a2_z, a3_z = guass_map_cal_delta(output_finite, (x_map_finite, y_map_finite, z_map_finite), mask)
            error_delta['mask_pix'].append(torch.sum(mask).item())
            error_delta['a1_x'].append(a1_x.item())
            error_delta['a2_x'].append(a2_x.item())
            error_delta['a3_x'].append(a3_x.item())

            error_delta['a1_y'].append(a1_y.item())
            error_delta['a2_y'].append(a2_y.item())
            error_delta['a3_y'].append(a3_y.item())

            error_delta['a1_z'].append(a1_z.item())
            error_delta['a2_z'].append(a2_z.item())
            error_delta['a3_z'].append(a3_z.item())

            error_x, error_y, error_z = sum_mean_acc(output_max, force_max, mask)
            error_force_map_max['error_x'].append(error_x)
            error_force_map_max['error_y'].append(error_y)
            error_force_map_max['error_z'].append(error_z)

    print('x error:{}'.format(np.mean(error_force_map_max['error_x'])))
    print('y error:{}'.format(np.mean(error_force_map_max['error_y'])))
    print('z error:{}'.format(np.mean(error_force_map_max['error_z'])))
    print('mean error:{}'.format((np.mean(error_force_map_max['error_x']) + np.mean(error_force_map_max['error_y']) + np.mean(error_force_map_max['error_z'])) / 3))

    print('a1_x:{}'.format(np.mean(error_delta['a1_x'])))
    print('a2_x:{}'.format(np.mean(error_delta['a2_x'])))
    print('a3_x:{}'.format(np.mean(error_delta['a3_x'])))

    print('a1_y:{}'.format(np.mean(error_delta['a1_y'])))
    print('a2_y:{}'.format(np.mean(error_delta['a2_y'])))
    print('a3_y:{}'.format(np.mean(error_delta['a3_y'])))

    print('a1_z:{}'.format(np.mean(error_delta['a1_z'])))
    print('a2_z:{}'.format(np.mean(error_delta['a2_z'])))
    print('a3_z:{}'.format(np.mean(error_delta['a3_z'])))
