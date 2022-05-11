import numpy as np
# np.set_printoptions(precision=3, suppress=True)
from models.backbone_rgbd_displacement_multi_robot import Backbone
from utils.load_data_rgbd_displacement_multi_robot import DMPDatasetEERandTarXYLang, pad_collate_xy_lang
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
import time
import random
import clip
import re
import json


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def pixel_position_to_attn_index(pixel_position, attn_map_offset=1):
    index = (pixel_position[:, 0]) // 8 + attn_map_offset + 28 * (pixel_position[:, 1] // 8)
    index = index.astype(int)
    index = torch.tensor(index).to(device).unsqueeze(1)
    return index


def attn_loss(attn_map, supervision, criterion, scale):
    # supervision = [[0, [-1]], [1, [1]], [2, [2]], [3, [3]], [4, [4]]]
    # supervision = [[1, [0, 2, 3]], [2, [2]], [4, [4]]]
    loss = 0
    for supervision_pair in supervision:
        target_attn = 0
        for i in supervision_pair[1]:
            target_attn = target_attn + attn_map[:, supervision_pair[0], i]
        loss = loss + criterion(target_attn, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device))
    loss = loss * scale
    return loss



def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, scheduler, criterion, ckpt_path, save_ckpt, stage,
    print_attention_map=False, curriculum_learning=False, supervised_attn=False):
    model.train()
    criterion2 = nn.L1Loss(reduction='none')

    for idx, (img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj, displacement) in enumerate(data_loader):
        global_step = epoch_idx * len(data_loader) + idx

        # Prepare data
        img = img.to(device)
        target = target.to(device)
        joint_angles = joint_angles.to(device)
        ee_pos = ee_pos.to(device)
        ee_traj = ee_traj.to(device)
        length = length.to(device)
        target_pos = target_pos.to(device)
        phis = phis.to(device)
        mask = mask.to(device)
        attn_index_tar = pixel_position_to_attn_index(target_xy, attn_map_offset=5)
        attn_index_ee = pixel_position_to_attn_index(ee_xy, attn_map_offset=5)
        sentence = sentence.to(device)
        joint_angles_traj = joint_angles_traj.to(device)
        ee_traj = torch.cat((ee_traj, joint_angles_traj[:, -1:, :]), axis=1)
        displacement = displacement.to(device)

        # print(img.shape)

        # mean = np.array([ 2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0])
        # std = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1]) ** (1/2)
        # print(ee_pos[0].detach().cpu().numpy() * std + mean)
        # print(target[0])
        # print(target_xy[0])
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 3, 1)
        # plt.imshow(img.detach().cpu().numpy()[0, :, :, :3])
        # cir = plt.Circle(target_xy[0], 5, color='r')
        # ax.add_patch(cir)

        # fig.add_subplot(1, 3, 2)
        # plt.imshow(img.detach().cpu().numpy()[0, :, :, 3])

        # ax3 = fig.add_subplot(1, 3, 3)
        # attn_map_target = np.zeros((28 * 28,))
        # attn_map_target[attn_index[0] - 2] = 1
        # attn_map_target = attn_map_target.reshape((28, 28))
        # plt.imshow(attn_map_target)


        # plt.show()

        # Forward pass
        optimizer.zero_grad()
        if stage == 0:
            target_position_pred, ee_pos_from_joints_pred, ee_pos_from_img_pred, attn_map, attn_map2 = model(img, joint_angles, sentence, phis, stage)
        elif stage == 1:
            target_position_pred, ee_pos_from_joints_pred, ee_pos_from_img_pred, displacement_pred, attn_map, attn_map2, attn_map3 = model(img, joint_angles, sentence, phis, stage)
        else:
            target_position_pred, ee_pos_from_joints_pred, ee_pos_from_img_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, phis, stage)

        loss0 = criterion(target_position_pred, target_pos)
        writer.add_scalar('train loss tar pos', loss0.item(), global_step=epoch_idx * len(data_loader) + idx)
        loss2 = criterion(ee_pos_from_joints_pred, ee_pos)
        writer.add_scalar('train loss ee pos from joints', loss2.item(), global_step=epoch_idx * len(data_loader) + idx)
        loss3 = criterion(ee_pos_from_img_pred, ee_pos)
        writer.add_scalar('train loss ee pos from img', loss3.item(), global_step=epoch_idx * len(data_loader) + idx)
        loss = loss0 + loss2 + loss3

        # Attention Supervision for layer1
        supervision_layer1 = [[0, [-1]], [1, [1]], [2, [2]], [3, [3]], [4, [4]]]
        loss_attn_layer1 = attn_loss(attn_map, supervision_layer1, criterion, scale=5000)

        # Attention Supervision for layer2
        supervision_layer2 = [[1, [1]], [2, [-2]], [4, [4]]]
        loss_attn_layer2 = attn_loss(attn_map2, supervision_layer2, criterion, scale=5000)
        
        # Attention Supervision for Target Pos
        target_pos_attn = torch.gather(attn_map2[:, 0, :], 1, attn_index_tar)
        loss_target_pos_attn = criterion(target_pos_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device)) * 5000

        # Attention Supervision for EE from img
        ee_img_attn = torch.gather(attn_map2[:, 3, :], 1, attn_index_ee)
        loss_ee_img_attn = criterion(ee_img_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device)) * 5000

        # Attention Loss
        loss_attn = loss_attn_layer1 + loss_attn_layer2 + loss_target_pos_attn + loss_ee_img_attn

        if stage >= 1:
            loss1 = criterion(displacement_pred, displacement)
            supervision_layer3 = [[1, [0, 2, 3]], [2, [2]], [4, [4]]]
            loss_attn_layer3 = attn_loss(attn_map3, supervision_layer3, criterion, scale=5000)
            loss = loss + loss1
            loss_attn = loss_attn + loss_attn_layer3

        if stage >= 2:
            # Attention Supervision for Target Pos, EEF Pos, Command
            traj_attn = attn_map4[:, 4, 1] + attn_map4[:, 4, 2] + attn_map4[:, 4, -1] + attn_map4[:, 4, -2]
            loss_traj_attn = criterion(traj_attn, torch.ones(attn_map4.shape[0], 1, dtype=torch.float32).to(device)) * 5000
            loss_attn = loss_attn + loss_traj_attn

            # Only training on xyz, ignoring rpy
            # For trajectory, use a pre-defined weight matrix to indicate the importance of the trajectory points
            trajectory_pred = trajectory_pred * mask
            ee_traj = ee_traj * mask
            weight_matrix = torch.tensor(np.array([1 ** i for i in range(ee_traj.shape[-1])]), dtype=torch.float32) + torch.tensor(np.array([0.9 ** i for i in range(ee_traj.shape[-1]-1, -1, -1)]), dtype=torch.float32)
            weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(1).repeat(ee_traj.shape[0], ee_traj.shape[1], 1).cuda()
            loss4 = (criterion2(trajectory_pred, ee_traj) * weight_matrix).sum() / (mask * weight_matrix).sum()
            writer.add_scalar('train loss traj', loss1.item(), global_step=epoch_idx * len(data_loader) + idx)
            loss = loss + loss4
            print('loss traj', loss4.item())

        loss = loss + loss_attn


        print(f'{loss_target_pos_attn.item():.2f}')
        print('obj1 pred', target_position_pred[0].detach().cpu().numpy())
        print('obj1 g_t_', target_pos[0].detach().cpu().numpy())
        print('obj2 pred', target_position_pred[1].detach().cpu().numpy())
        print('obj2 g_t_', target_pos[1].detach().cpu().numpy())

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss.item(), global_step=epoch_idx * len(data_loader) + idx)
        if stage == 0:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l0 {loss0.item():.2f}, l2 {loss2.item():.2f}, l3 {loss3.item():.2f}')
        elif stage == 1:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l0 {loss0.item():.2f}, l1 {loss1.item():.2f}, l2 {loss2.item():.2f}, l3 {loss3.item():.2f}')
        else:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l0 {loss0.item():.2f}, l1 {loss1.item():.2f}, l2 {loss2.item():.2f}, l3 {loss3.item():.2f}, l4 {loss4.item():.2f}')

        # Print Attention Map
        if print_attention_map:
         
            print(target[0])
            print(target_xy[0])
            print(target_pos[0])
            attn_map = np.zeros((785,))
            attn_map[attn_index[0]] = 1
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(attn_map[1:785].reshape((28, 28)))
            # plt.colorbar()
            fig.add_subplot(1, 2, 2)
            plt.imshow(img.detach().cpu().numpy()[0])
            plt.show()

        # Save checkpoint
        if save_ckpt:
            if not os.path.isdir(os.path.join(ckpt_path, name)):
                os.mkdir(os.path.join(ckpt_path, name))
            if global_step % 5000 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, name, f'{global_step}.pth'))

        # if global_step == 50:
        #     scheduler.step()

        # elif global_step == 100:
        #     scheduler.step()
    return stage


def test(writer, name, epoch_idx, data_loader, model, criterion, train_dataset_size, stage, print_attention_map=False, train_split=False):
    with torch.no_grad():
        model.eval()
        error_trajectory = 0
        error_gripper = 0
        loss5_accu = 0
        idx = 0
        error_target_position = 0
        error_displacement = 0
        error_ee_pos = 0
        error_joints_prediction = 0
        num_datapoints = 0
        num_trajpoints = 0
        num_grippoints = 0
        criterion2 = nn.MSELoss(reduction='none')

        mean = np.array([ 2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0])
        std = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1]) ** (1/2)
        mean_joints = np.array([-0.00357743, 0.29354134, 0.03703507, -2.01260356, -0.03319358, 0.76566389, 0.05069619, 0.01733641])
        std_joints = np.array([0.07899751, 0.04528939, 0.27887484, 0.10307656, 0.06242473, 0.04195134, 0.27607541, 0.00033524]) ** (1/2)
        mean_traj_gripper = np.array([2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0, 0.01733641])
        std_traj_gripper = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1, 0.00033524]) ** (1/2)
        mean_displacement = np.array([2.53345831e-01, 1.14758266e-01, -6.98193015e-02, 0, 0, 0, 0, 0, 0])
        std_displacement = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])
        # std_traj_gripper_centered = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1, 0.23799407366571126])
        
        for idx, (img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj, displacement) in enumerate(data_loader):
            global_step = epoch_idx * len(data_loader) + idx

            # Prepare data
            img = img.to(device)
            target = target.to(device)
            joint_angles = joint_angles.to(device)
            ee_pos = ee_pos.to(device)
            ee_traj = ee_traj.to(device)
            length = length.to(device)
            target_pos = target_pos.to(device)
            phis = phis.to(device)
            mask = mask.to(device)
            sentence = sentence.to(device)
            joint_angles_traj = joint_angles_traj.to(device)
            ee_traj = torch.cat((ee_traj, joint_angles_traj[:, -1:, :]), axis=1)

            # Forward pass
            if stage == 0:
                target_position_pred, ee_pos_from_joints_pred, ee_pos_from_img_pred, attn_map, attn_map2 = model(img, joint_angles, sentence, phis, stage)
            elif stage == 1:
                target_position_pred, ee_pos_from_joints_pred, ee_pos_from_img_pred, displacement_pred, attn_map, attn_map2, attn_map3 = model(img, joint_angles, sentence, phis, stage)
            else:
                target_position_pred, ee_pos_from_joints_pred, ee_pos_from_img_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, phis, stage)


            target_pos = target_pos.detach().cpu()
            target_position_pred = target_position_pred.detach().cpu()
            error_target_position_this_time = torch.sum(((target_position_pred[:, :3] - target_pos[:, :3]) * torch.tensor(std[:3])) ** 2, axis=1) ** 0.5
            error_target_position += error_target_position_this_time.sum()
            num_datapoints += error_target_position_this_time.shape[0]

            ee_pos = ee_pos.detach().cpu()
            ee_pos_from_joints_pred = ee_pos_from_joints_pred.detach().cpu()
            error_ee_pos_this_time = torch.sum(((ee_pos_from_joints_pred[:, :3] - ee_pos[:, :3]) * torch.tensor(std[:3])) ** 2, axis=1) ** 0.5
            error_ee_pos += error_ee_pos_this_time.sum()

            if stage >= 1:
                displacement_pred = displacement_pred.detach().cpu()
                error_displace_this_time = torch.sum(((displacement_pred[:, :3] - displacement[:, :3]) * torch.tensor(std_displacement[:3])) ** 2, axis=1) ** 0.5
                error_displacement += error_displace_this_time.sum()

            if stage >= 2:
                trajectory_pred = trajectory_pred * mask
                ee_traj = ee_traj * mask
                # Only training on xyz, ignoring rpy
                # loss1 = criterion2(trajectory_pred, ee_traj).sum() / mask.sum()

                trajectory_pred = trajectory_pred.detach().cpu().transpose(2, 1)
                ee_traj = ee_traj.detach().cpu().transpose(2, 1)
                target_pos = target_pos.detach().cpu()
                
                error_trajectory_this_time = torch.sum(((trajectory_pred[:, :, :3] - ee_traj[:, :, :3]) * torch.tensor(std[:3])) ** 2, axis=2) ** 0.5
                error_trajectory_this_time = torch.sum(error_trajectory_this_time)
                error_trajectory += error_trajectory_this_time
                num_trajpoints += torch.sum(mask[:, :3, :]) / mask.shape[1]

                error_gripper_this_time = torch.sum(((trajectory_pred[:, :, 3:] - ee_traj[:, :, 3:]) * torch.tensor([std_joints[-1]])) ** 2, axis=2) ** 0.5
                error_gripper_this_time = torch.sum(error_gripper_this_time)
                error_gripper += error_gripper_this_time
                num_grippoints += torch.sum(mask[:, 3, :]) / mask.shape[1]

            # Print Attention Map
            if print_attention_map:
                if stage > 1:
                    trajectory_pred = trajectory_pred * std_traj_gripper
                    target_position_pred = target_position_pred * std
                    target_pos = target_pos * std
                    ee_traj = ee_traj * std_traj_gripper
                    gripper = (joint_angles_traj[0, -1, :].detach().cpu() * std_traj_gripper[-1]).numpy()
                    gripper_pred = trajectory_pred[0, :, 9].detach().cpu().numpy()
                    gripper_x = np.arange(len(gripper))

                    fig = plt.figure(num=1, clear=True)
                    # ax = fig.add_subplot(1, 3, 1, projection='3d')
                    # x_ee = trajectory_pred[0, :, 0].detach().cpu().numpy()
                    # y_ee = trajectory_pred[0, :, 1].detach().cpu().numpy()
                    # z_ee = trajectory_pred[0, :, 2].detach().cpu().numpy()
                    # x_target = target_position_pred[0, 0].detach().cpu().numpy()
                    # y_target = target_position_pred[0, 1].detach().cpu().numpy()
                    # z_target = target_position_pred[0, 2].detach().cpu().numpy()
                    # x_target_gt = target_pos[0, 0].detach().cpu().numpy()
                    # y_target_gt = target_pos[0, 1].detach().cpu().numpy()
                    # z_target_gt = target_pos[0, 2].detach().cpu().numpy()
                    # x_ee_gt = ee_traj[0, :, 0].detach().cpu().numpy()
                    # y_ee_gt = ee_traj[0, :, 1].detach().cpu().numpy()
                    # z_ee_gt = ee_traj[0, :, 2].detach().cpu().numpy()
                    # ax.scatter3D(x_ee, y_ee, z_ee, color='green')
                    # ax.scatter3D(x_target, y_target, z_target, color='blue')
                    # ax.scatter3D(x_target_gt, y_target_gt, z_target_gt, color='red')
                    # ax.scatter3D(x_ee_gt, y_ee_gt, z_ee_gt, color='grey')

                    ax = fig.add_subplot(1, 2, 1)
                    ax.imshow(attn_map2[0, 0, 5:5+28*28].detach().cpu().numpy().reshape((28, 28))[::-1, :])

                    ax = fig.add_subplot(1, 2, 2)
                    ax.imshow(img[0, :, :, :3].detach().cpu().numpy()[::-1, :, :])


                    # ax = fig.add_subplot(1, 3, 3)
                    # ax.plot(gripper_x, gripper)
                    # ax.plot(gripper_x, gripper_pred)

                    plt.show()

                    # save_name = name
                    # if train_split:
                    #     save_name = save_name + '_train_split'
                    # if not os.path.isdir(f'results_png/'):
                    #     os.mkdir(f'results_png/')
                    # if not os.path.isdir(f'results_png/{save_name}/'):
                    #     os.mkdir(f'results_png/{save_name}/')
                    # if not os.path.isdir(f'results_png/{save_name}/{epoch_idx}/'):
                    #     os.mkdir(f'results_png/{save_name}/{epoch_idx}/')
                    # plt.savefig(os.path.join(f'results_png/{save_name}/{epoch_idx}/', f'{idx}.png'))


            idx += 1

            # Print
            # print(f'test: epoch {epoch_idx}, step {idx}, loss5 {loss5.item():.2f}')
            if stage == 0:
                print(idx, 'err tar pos:', error_target_position / num_datapoints, 'err ee pos:', error_ee_pos / num_datapoints)
            elif stage == 1:
                print(idx, 'err tar pos:', (error_target_position / num_datapoints).item(), 'err ee pos:', (error_ee_pos / num_datapoints).item(), 'err displace:', (error_displacement / num_datapoints).item())
            else:
                print(idx, f'err tar pos: {(error_target_position / num_datapoints).item():.4f} err ee pos: {(error_ee_pos / num_datapoints).item():.4f} err displace: {(error_displacement / num_datapoints).item():.4f}')
                print(idx, f'err traj {(error_trajectory / num_trajpoints).item():.4f} err grip {(error_gripper / num_grippoints).item():.4f}')

        # Log
        if writer is not None:
            if not train_split:
                writer.add_scalar('test error_target_position', error_target_position / num_datapoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('test error_ee_pos', error_ee_pos / num_datapoints, global_step=epoch_idx * train_dataset_size)
                if stage > 1:
                    writer.add_scalar('test error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)
                    writer.add_scalar('test error_gripper', error_gripper / num_grippoints, global_step=epoch_idx * train_dataset_size)
            else:
                writer.add_scalar('train_split error_target_position', error_target_position / num_datapoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('train_split error_ee_pos', error_ee_pos / num_datapoints, global_step=epoch_idx * train_dataset_size)
                if stage > 1:
                    writer.add_scalar('train_split error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)
                    writer.add_scalar('train_split error_gripper', error_gripper / num_grippoints, global_step=epoch_idx * train_dataset_size)

        print((error_target_position / num_datapoints).item())
        print((error_ee_pos / num_datapoints).item())
        print((error_trajectory / num_trajpoints).item())
        print((error_gripper / num_grippoints).item())

        return {
            'error_target_position': (error_target_position / num_datapoints).item(),
            'error_ee_pos': (error_ee_pos / num_datapoints).item(),
            'error_trajectory': (error_trajectory / num_trajpoints).item(),
            'error_gripper': (error_gripper / num_grippoints).item(),
        }


def test_ckpt(ckpt_path, name, ckpt, data_loaders):

    ckpt_folder = os.path.join(ckpt_path, name)
    ckpt_file = os.path.join(ckpt_folder, ckpt)
    pretrained_dict = torch.load(os.path.join(ckpt_file))['model']

    print('loaded', ckpt)
    criterion = nn.MSELoss()

    results = {}
    for dataloader_name in data_loaders:
        if dataloader_name == 'ur5':
            model = Backbone(img_size=224, embedding_size=192, num_traces_in=7, num_traces_out=10, num_weight_points=12)
        else:
            model = Backbone(img_size=224, embedding_size=192, num_traces_in=8, num_traces_out=10, num_weight_points=12)

        generic_re = re.compile('|'.join(model.do_not_load))
        pretrained_dict = {k:pretrained_dict[k] for k in pretrained_dict if not re.match(generic_re, k)}
        pretrained_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = False

        model = model.to(device)

        result = test(None, name, 0, data_loaders[dataloader_name], model, criterion, 0, stage=2, print_attention_map=True)
        results[dataloader_name] = result
    return results


def main(batch_size=256):
    # data_root_path = r'/data/Documents/yzhou298'
    data_root_path = r'/share/yzhou298'
    # data_root_path = r'/mnt/disk1'
    ckpt_path = os.path.join(data_root_path, r'ckpts/')
    save_ckpt = True
    supervised_attn = True
    curriculum_learning = True
    ckpt = None


    name = 'train-12-rgbd-mse-displacement-lr-1e-4-aligned-train-test-centered'
    ckpts = []
    for file in os.listdir(os.path.join(ckpt_path, name)):
        if file.endswith(".pth"):
            ckpts.append(file)


    # load data
    dataset_test_dmp = DMPDatasetEERandTarXYLang([os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224_test/')], random=False, length_total=120)
    data_loader_test_dmp = torch.utils.data.DataLoader(dataset_test_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    # dataset_test_dmp_panda = DMPDatasetEERandTarXYLang([os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224_panda_2/')], random=False, length_total=120, normalize='panda')
    # data_loader_test_dmp_panda = torch.utils.data.DataLoader(dataset_test_dmp_panda, batch_size=batch_size,
    #                                       shuffle=True, num_workers=8,
    #                                       collate_fn=pad_collate_xy_lang)

    print(ckpts)

    data_loaders = {
        'ur5': data_loader_test_dmp,
        # 'panda': data_loader_test_dmp_panda
    }

    results = {}
    for ckpt in ckpts:
        result = test_ckpt(ckpt_path, name, ckpt, data_loaders)
        results[int(ckpt.split(r'.')[0])] = result

        with open('results_panda_aligned_depth_and_light.json', 'w') as fp:
            json.dump(results, fp, indent=4)




if __name__ == '__main__':
    main()