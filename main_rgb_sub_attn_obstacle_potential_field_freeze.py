import numpy as np
# np.set_printoptions(precision=3, suppress=True)
from models.backbone_rgbd_sub_attn_obstacle_potential_field_freeze import Backbone
from utils.load_data_rgb_abs_action_fast_gripper_finetuned_attn_obstacle import DMPDatasetEERandTarXYLang, pad_collate_xy_lang
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

    for idx, (img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj, displacement, obstacle_pos, obstacle_xy) in enumerate(data_loader):
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
        attn_index_tar = pixel_position_to_attn_index(target_xy, attn_map_offset=6)
        attn_index_ee = pixel_position_to_attn_index(ee_xy, attn_map_offset=6)
        attn_index_obs = pixel_position_to_attn_index(obstacle_xy, attn_map_offset=6)
        sentence = sentence.to(device)
        joint_angles_traj = joint_angles_traj.to(device)
        ee_traj = torch.cat((ee_traj, joint_angles_traj[:, -1:, :]), axis=1)
        displacement = displacement.to(device)
        obstacle_pos = obstacle_pos.to(device)

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
            attn_map, attn_map2, obstacle_pred = model(img, joint_angles, sentence, phis, stage)
        elif stage == 1:
            target_position_pred, ee_pos_pred, displacement_pred, obstacle_pred, potential_vector_pred, attn_map, attn_map2, attn_map3 = model(img, joint_angles, sentence, phis, stage)
        else:
            target_position_pred, ee_pos_pred, displacement_pred, obstacle_pred, potential_vector_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, phis, stage)


        # Attention Supervision for layer1
        supervision_layer1 = [[0, [-1]], [1, [1]], [2, [2]], [3, [3]], [4, [4]]]
        loss_attn_layer1 = attn_loss(attn_map, supervision_layer1, criterion, scale=5000)
        reverse_supervision_layer1 = [[0, [5]], [1, [5]], [2, [5]], [3, [5]], [4, [5]]]
        reverse_loss_attn_layer1 = attn_loss(attn_map, reverse_supervision_layer1, criterion, scale=5000) - 5000 * 5

        # Attention Supervision for layer2
        supervision_layer2 = [[1, [1]], [2, [-2]], [4, [4]]]
        loss_attn_layer2 = attn_loss(attn_map2, supervision_layer2, criterion, scale=5000)
        reverse_supervision_layer2 = [[0, [5]], [1, [5]], [2, [5]], [3, [5]], [4, [5]]]
        reverse_loss_attn_layer2 = attn_loss(attn_map2, reverse_supervision_layer2, criterion, scale=5000) - 5000 * 5
        
        # # Attention Supervision for Target Pos
        # target_pos_attn = torch.gather(attn_map2[:, 0, :], 1, attn_index_tar)
        # loss_target_pos_attn = criterion(target_pos_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device)) * 5000

        # # Attention Supervision for EE from img
        # ee_img_attn = torch.gather(attn_map2[:, 3, :], 1, attn_index_ee)
        # loss_ee_img_attn = criterion(ee_img_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device)) * 5000

        # Attention Supervision for obstacle from img
        obs_img_attn = torch.gather(attn_map2[:, 5, :], 1, attn_index_obs)
        loss_obs_img_attn = criterion(obs_img_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device)) * 5000

        # Aux task loss for obstacle pos prediction
        loss5 = criterion(obstacle_pred, obstacle_pos)

        # Attention Loss
        loss_attn = loss_attn_layer1 + loss_attn_layer2 - reverse_loss_attn_layer1 - reverse_loss_attn_layer2 + loss_obs_img_attn
        loss = loss5

        writer.add_scalar('train/loss obstacle pos', loss5.item(), global_step=epoch_idx * len(data_loader) + idx)
        writer.add_scalar('train_attn/loss attn layer1', loss_attn_layer1.item(), global_step=epoch_idx * len(data_loader) + idx)
        writer.add_scalar('train_attn/loss attn layer2', loss_attn_layer2.item(), global_step=epoch_idx * len(data_loader) + idx)
        writer.add_scalar('train_attn/reverse loss attn layer1', 0 - reverse_loss_attn_layer1.item(), global_step=epoch_idx * len(data_loader) + idx)
        writer.add_scalar('train_attn/reverse loss attn layer2', 0 - reverse_loss_attn_layer2.item(), global_step=epoch_idx * len(data_loader) + idx)
        writer.add_scalar('train_attn/loss_obs_img_attn', loss_obs_img_attn.item(), global_step=epoch_idx * len(data_loader) + idx)

        if stage >= 1:
            loss0 = criterion(target_position_pred, target_pos)
            loss1 = criterion(displacement_pred, displacement)
            loss2 = criterion(ee_pos_pred, ee_pos)
            loss6 = criterion(potential_vector_pred, ee_pos[:, :3] - obstacle_pos)

            # Attention Supervision for layer 3
            supervision_layer3 = [[0, [0]], [1, [0, 2, 3]], [2, [2, 3]], [4, [4]], [5, [2, 3, 5]]]
            loss_attn_layer3 = attn_loss(attn_map3, supervision_layer3, criterion, scale=5000)
            reverse_supervision_layer3 = [[0, [5]], [1, [5]], [2, [5]], [3, [5]], [4, [5]]]
            reverse_loss_attn_layer3 = attn_loss(attn_map3, reverse_supervision_layer3, criterion, scale=5000) - 5000 * 5


            writer.add_scalar('train/loss tar pos', loss0.item(), global_step=epoch_idx * len(data_loader) + idx)
            writer.add_scalar('train/displacement', loss1.item(), global_step=epoch_idx * len(data_loader) + idx)
            writer.add_scalar('train/loss ee pos from joints', loss2.item(), global_step=epoch_idx * len(data_loader) + idx)
            writer.add_scalar('train/loss potential vector', loss6.item(), global_step=epoch_idx * len(data_loader) + idx)
            writer.add_scalar('train_attn/reverse loss attn layer3', 0 - reverse_loss_attn_layer3.item(), global_step=epoch_idx * len(data_loader) + idx)
            writer.add_scalar('train_attn/loss attn layer3', loss_attn_layer3.item(), global_step=epoch_idx * len(data_loader) + idx)

            loss = loss5 + loss6
            loss_attn = loss_attn + loss_attn_layer3 - reverse_loss_attn_layer3

            # print(f'{loss_target_pos_attn.item():.2f}')
            # print('obj1 pred', target_position_pred[0].detach().cpu().numpy())
            # print('obj1 g_t_', target_pos[0].detach().cpu().numpy())
            # print('obj2 pred', target_position_pred[1].detach().cpu().numpy())
            # print('obj2 g_t_', target_pos[1].detach().cpu().numpy())

        if stage >= 2:
            # Attention Supervision for Target Pos, EEF Pos, Command
            traj_attn = attn_map4[:, 4, 0] + attn_map4[:, 4, 1] + attn_map4[:, 4, 2] + attn_map4[:, 4, 5] + attn_map4[:, 4, -1] + attn_map4[:, 4, -2]
            loss_traj_attn = criterion(traj_attn, torch.ones(attn_map4.shape[0], 1, dtype=torch.float32).to(device)) * 5000
            loss_attn = loss_attn + loss_traj_attn

            # Only training on xyz, ignoring rpy
            # For trajectory, use a pre-defined weight matrix to indicate the importance of the trajectory points
            trajectory_pred = trajectory_pred * mask
            ee_traj = ee_traj * mask
            weight_matrix = torch.tensor(np.array([1 ** i for i in range(ee_traj.shape[-1])]), dtype=torch.float32) + torch.tensor(np.array([0.9 ** i for i in range(ee_traj.shape[-1]-1, -1, -1)]), dtype=torch.float32)
            weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(1).repeat(ee_traj.shape[0], ee_traj.shape[1], 1).cuda()
            loss4 = (criterion2(trajectory_pred, ee_traj) * weight_matrix).sum() / (mask * weight_matrix).sum()
            writer.add_scalar('train/loss traj', loss4.item(), global_step=epoch_idx * len(data_loader) + idx)
            writer.add_scalar('train_attn/loss attn layer4', loss_traj_attn.item(), global_step=epoch_idx * len(data_loader) + idx)
            loss = loss + loss4
            print('loss traj', loss4.item())

        loss = loss + loss_attn


        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss.item(), global_step=epoch_idx * len(data_loader) + idx)
        if stage == 0:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}')
        elif stage == 1:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l0 {loss0.item():.2f}, l1 {loss1.item():.2f}, l2 {loss2.item():.2f}, l5 {loss5.item():.2f}')
        else:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l0 {loss0.item():.2f}, l1 {loss1.item():.2f}, l2 {loss2.item():.2f}, l4 {loss4.item():.2f}, l5 {loss5.item():.2f}')

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
            if global_step % 10000 == 0:
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
        mean_joints = np.array([-2.26736831e-01, 5.13238925e-01, -1.84928474e+00, 7.77270127e-01, 1.34229937e+00, 1.39107280e-03, 2.12295943e-01])
        std_joints = np.array([1.41245676e-01, 3.07248648e-02, 1.34113984e-01, 6.87947763e-02, 1.41992804e-01, 7.84910314e-05, 5.66411791e-02]) ** (1/2)
        mean_traj_gripper = np.array([2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0, 2.12295943e-01])
        std_traj_gripper = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1, 5.66411791e-02]) ** (1/2)
        mean_displacement = np.array([2.53345831e-01, 1.14758266e-01, -6.98193015e-02, 0, 0, 0, 0, 0, 0])
        std_displacement = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])
        std_traj_gripper_centered = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1, 0.23799407366571126])
        
        for idx, (img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj, displacement, obstacle_pos, obstacle_xy) in enumerate(data_loader):
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
            obstacle_pos = obstacle_pos.to(device)
            attn_index_obs = pixel_position_to_attn_index(obstacle_xy, attn_map_offset=6)

            # Forward pass
            if stage == 0:
                obstacle_pred, attn_map, attn_map2 = model(img, joint_angles, sentence, phis, stage)
            elif stage == 1:
                target_position_pred, ee_pos_pred, displacement_pred, obstacle_pred, potential_vector_pred, attn_map, attn_map2, attn_map3 = model(img, joint_angles, sentence, phis, stage)
            else:
                target_position_pred, ee_pos_pred, displacement_pred, obstacle_pred, potential_vector_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, phis, stage)



            if stage >= 1:
                target_pos = target_pos.detach().cpu()
                target_position_pred = target_position_pred.detach().cpu()
                error_target_position_this_time = torch.sum(((target_position_pred[:, :3] - target_pos[:, :3]) * torch.tensor(std[:3])) ** 2, axis=1) ** 0.5
                error_target_position += error_target_position_this_time.sum()
                num_datapoints += error_target_position_this_time.shape[0]

                ee_pos = ee_pos.detach().cpu()
                ee_pos_pred = ee_pos_pred.detach().cpu()
                error_ee_pos_this_time = torch.sum(((ee_pos_pred[:, :3] - ee_pos[:, :3]) * torch.tensor(std[:3])) ** 2, axis=1) ** 0.5
                error_ee_pos += error_ee_pos_this_time.sum()

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
                
                error_trajectory_this_time = torch.sum(((trajectory_pred[:, :, :3] - ee_traj[:, :, :3]) * torch.tensor(std_displacement[:3])) ** 2, axis=2) ** 0.5
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
                    trajectory_pred = trajectory_pred * std_traj_gripper_centered
                    target_position_pred = target_position_pred * std
                    target_pos = target_pos * std
                    ee_traj = ee_traj * std_traj_gripper_centered
                    gripper = (joint_angles_traj[0, -1, :].detach().cpu() * std_traj_gripper[-1]).numpy()
                    gripper_pred = trajectory_pred[0, :, 9].detach().cpu().numpy()
                    gripper_x = np.arange(len(gripper))

                    fig = plt.figure(num=1, clear=True)
                    ax = fig.add_subplot(1, 3, 1, projection='3d')
                    x_ee = trajectory_pred[0, :, 0].detach().cpu().numpy()
                    y_ee = trajectory_pred[0, :, 1].detach().cpu().numpy()
                    z_ee = trajectory_pred[0, :, 2].detach().cpu().numpy()
                    x_target = target_position_pred[0, 0].detach().cpu().numpy()
                    y_target = target_position_pred[0, 1].detach().cpu().numpy()
                    z_target = target_position_pred[0, 2].detach().cpu().numpy()
                    x_target_gt = target_pos[0, 0].detach().cpu().numpy()
                    y_target_gt = target_pos[0, 1].detach().cpu().numpy()
                    z_target_gt = target_pos[0, 2].detach().cpu().numpy()
                    x_ee_gt = ee_traj[0, :, 0].detach().cpu().numpy()
                    y_ee_gt = ee_traj[0, :, 1].detach().cpu().numpy()
                    z_ee_gt = ee_traj[0, :, 2].detach().cpu().numpy()
                    ax.scatter3D(x_ee, y_ee, z_ee, color='green')
                    ax.scatter3D(x_target, y_target, z_target, color='blue')
                    ax.scatter3D(x_target_gt, y_target_gt, z_target_gt, color='red')
                    ax.scatter3D(x_ee_gt, y_ee_gt, z_ee_gt, color='grey')

                    ax = fig.add_subplot(1, 3, 2)
                    ax.imshow(img[0, :, :, :3].detach().cpu().numpy()[::-1, :, :])


                    ax = fig.add_subplot(1, 3, 3)
                    ax.plot(gripper_x, gripper)
                    ax.plot(gripper_x, gripper_pred)

                    # plt.show()

                    save_name = name
                    if train_split:
                        save_name = save_name + '_train_split'
                    if not os.path.isdir(f'results_png/'):
                        os.mkdir(f'results_png/')
                    if not os.path.isdir(f'results_png/{save_name}/'):
                        os.mkdir(f'results_png/{save_name}/')
                    if not os.path.isdir(f'results_png/{save_name}/{epoch_idx}/'):
                        os.mkdir(f'results_png/{save_name}/{epoch_idx}/')
                    plt.savefig(os.path.join(f'results_png/{save_name}/{epoch_idx}/', f'{idx}.png'))

            idx += 1

            # Print
            # print(f'test: epoch {epoch_idx}, step {idx}, loss5 {loss5.item():.2f}')
            if stage >= 1:
                print(idx, f'err tar pos: {(error_target_position / num_datapoints).item():.4f} err ee pos: {(error_ee_pos / num_datapoints).item():.4f} err displace: {(error_displacement / num_datapoints).item():.4f}')
            if stage >= 2:
                print(idx, f'err traj {(error_trajectory / num_trajpoints).item():.4f} err grip {(error_gripper / num_grippoints).item():.4f}')

        # Log
        if not train_split:
            if stage >= 1:
                writer.add_scalar('test/error_target_position', error_target_position / num_datapoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('test/error_ee_pos', error_ee_pos / num_datapoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('test/error_displace', error_displacement / num_datapoints, global_step=epoch_idx * train_dataset_size)
            if stage >= 2:
                writer.add_scalar('test/error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('test/error_gripper', error_gripper / num_grippoints, global_step=epoch_idx * train_dataset_size)
        else:
            if stage >= 1:
                writer.add_scalar('train_split/error_target_position', error_target_position / num_datapoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('train_split/error_ee_pos', error_ee_pos / num_datapoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('train_split/error_displace', error_displacement / num_datapoints, global_step=epoch_idx * train_dataset_size)
            if stage >= 2:
                writer.add_scalar('train_split/error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)
                writer.add_scalar('train_split/error_gripper', error_gripper / num_grippoints, global_step=epoch_idx * train_dataset_size)


def main(writer, name, batch_size=96):
    # data_root_path = r'/data/Documents/yzhou298'
    # data_root_path = r'/share/yzhou298'
    data_root_path = r'../'
    ckpt_path = os.path.join(data_root_path, r'ckpts/')
    save_ckpt = True
    supervised_attn = True
    curriculum_learning = True
    ckpt = '../train-12-rgb-sub-attn-fast-gripper-abs-action/440000.pth'

    # load model
    model = Backbone(img_size=224, embedding_size=192, num_traces_in=7, num_traces_out=10, num_weight_points=12, input_nc=3)
    if ckpt is not None:
        state_dict = torch.load(ckpt)['model']
        seg_embed = state_dict['seg_embed1.weight']
        add_on_embed = torch.zeros(1, seg_embed.shape[1], dtype=torch.float32).to(device)
        seg_embed = torch.cat((seg_embed, add_on_embed), axis=0)
        state_dict['seg_embed_1.weight'] = seg_embed
        print(seg_embed.shape)

        # if the checkpoint does not include the seg_embed_obstacle, we need to initialize it. 
        if 'seg_embed_obs.weight' not in state_dict:
            seg_embed_obs = nn.Embedding(1, seg_embed.shape[1]).weight.clone().detach().requires_grad_(True).to(device)
            seg_embed_obs = torch.cat((torch.zeros(seg_embed.shape[0] - 1, seg_embed.shape[1]).to(device), seg_embed_obs), axis=0)
            # seg_embed_obs = torch.zeros(seg_embed.shape[0], seg_embed.shape[1])
            state_dict['seg_embed_obs.weight'] = seg_embed_obs
        # print(state_dict['obstacle_slot'].shape, type(state_dict['obstacle_slot']))
        print(state_dict['tar_pos_slot'].shape, type(state_dict['tar_pos_slot']))
        # state_dict['obstacle_slot'] = torch.zeros(192).to(device)
        model.load_state_dict(state_dict, strict=False)
    # Freeze the model
    for para in model.parameters():
        para.requires_grad = False
    # Only unfreeze the needed part: obstacle token, obstacle token embeding, embed_to_obstacle_pos, embed_to_potential_field, controller
    model.obstacle_slot.requires_grad = True
    for para in model.requests_to_queries[-1].parameters():
        para.requires_grad = True
    for para in model.requests_to_keys[-1].parameters():
        para.requires_grad = True
    for para in model.requests_to_values[-1].parameters():
        para.requires_grad = True

    for para in model.seg_embed_obs.parameters():
        para.requires_grad = True
    for para in model.embed_to_obstacle_pos.parameters():
        para.requires_grad = True
    for para in model.controller.parameters():
        para.requires_grad = True
    for para in model.embed_to_potential_field.parameters():
        para.requires_grad = True

    model = model.to(device)

    # load data
    data_dirs = [
        os.path.join(data_root_path, 'mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224_obstacle_100/'),
        os.path.join(data_root_path, 'mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224_obstacle_not_in_the_way_100/'),
    ]
    data_dirs_test = [
        os.path.join(data_root_path, 'mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224_obstacle_val/'),
    ]
    dataset_train = DMPDatasetEERandTarXYLang(data_dirs, random=True, length_total=120)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    dataset_test = DMPDatasetEERandTarXYLang(data_dirs_test, random=True, length_total=120)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    # dataset_train_split = DMPDatasetEERandTarXYLang([os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224_train_split/')], random=True, length_total=120)
    # data_loader_train_split = torch.utils.data.DataLoader(dataset_train_split, batch_size=batch_size,
    #                                       shuffle=True, num_workers=8,
    #                                       collate_fn=pad_collate_xy_lang)
    
    dataset_train_dmp = DMPDatasetEERandTarXYLang(data_dirs, random=False, length_total=120)
    data_loader_train_dmp = torch.utils.data.DataLoader(dataset_train_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    dataset_test_dmp = DMPDatasetEERandTarXYLang(data_dirs_test, random=False, length_total=120)
    data_loader_test_dmp = torch.utils.data.DataLoader(dataset_test_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    # dataset_train_split_dmp = DMPDatasetEERandTarXYLang([os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224_train_split/')], random=False, length_total=120)
    # data_loader_train_split_dmp = torch.utils.data.DataLoader(dataset_train_split_dmp, batch_size=batch_size,
    #                                       shuffle=True, num_workers=8,
    #                                       collate_fn=pad_collate_xy_lang)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    print('loaded')

    # train n epoches
    loss_stage = 0
    for i in range(0, 300):
        whether_test = (((i + 1) % 10) == 0)
        if loss_stage < 0:
            loss_stage = train(writer, name, i, data_loader_train, model, optimizer, scheduler,
                criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=False)
            if whether_test:
                test(writer, name, i + 1, data_loader_test, model, criterion, len(data_loader_train), loss_stage, print_attention_map=False)
                # test(writer, name, i + 1, data_loader_train_split, model, criterion, len(data_loader_train), loss_stage, print_attention_map=True, train_split=True)
        else:
            loss_stage = train(writer, name, i, data_loader_train_dmp, model, optimizer, scheduler,
                criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=False)
            if whether_test:
                test(writer, name, i + 1, data_loader_test_dmp, model, criterion, len(data_loader_train_dmp), loss_stage, print_attention_map=False)
                # test(writer, name, i + 1, data_loader_train_split_dmp, model, criterion, len(data_loader_train_dmp), loss_stage, print_attention_map=True, train_split=True)
        if i >= 1 and i <= 2:
            loss_stage = 1
        elif i > 2:
            loss_stage = 2
        # if i > 2:
        #    loss_stage = 2


if __name__ == '__main__':
    name = 'train-12-rgb-sub-attn-fast-gripper-abs-action-obstacle-mini-part-not-in-the-way-potential-freeze-debug'
    writer = SummaryWriter('runs/' + name)
    main(writer, name)
