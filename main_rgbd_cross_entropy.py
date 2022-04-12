import numpy as np
# np.set_printoptions(precision=3, suppress=True)
from models.backbone_rgbd import Backbone
from utils.load_data_rgbd import DMPDatasetEERandTarXYLang, pad_collate_xy_lang
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
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

# https://stackoverflow.com/questions/68609414/how-to-calculate-correct-cross-entropy-between-2-tensors-in-pytorch-when-target
def cross_entropy(pred, target, eps=1e-5):
    return torch.mean(-torch.sum(target * torch.log(pred + eps), 1))


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, scheduler, criterion, ckpt_path, save_ckpt, stage,
    print_attention_map=False, curriculum_learning=False, supervised_attn=False):
    model.train()
    criterion2 = nn.L1Loss(reduction='none')

    for idx, (img, target, joint_angles, ee_pos, ee_traj, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj) in enumerate(data_loader):
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
        attn_index = pixel_position_to_attn_index(target_xy, attn_map_offset=2)
        sentence = sentence.to(device)
        joint_angles_traj = joint_angles_traj.to(device)
        ee_traj = torch.cat((ee_traj, joint_angles_traj[:, -1:, :]), axis=1)

        # Forward pass
        optimizer.zero_grad()
        if stage == 0:
            target_position_pred, attn_map, attn_map2 = model(img, ee_pos, sentence, phis, stage)
        else:
            target_position_pred, attn_map, attn_map2, attn_map3, trajectory_pred = model(img, ee_pos, sentence, phis, stage)

        loss5 = criterion(target_position_pred, target_pos)
        writer.add_scalar('train loss tar pos', loss5.item(), global_step=epoch_idx * len(data_loader) + idx)
        loss = loss5

        # Attention Supervision for target id
        # target_attn = attn_map[:, 0, -1]
        # loss_attn_target = criterion(target_attn, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device))
        target_attn_gt = torch.zeros(attn_map.shape[0], attn_map.shape[2], dtype=torch.float32).to(device)
        target_attn_gt[:, -1] = 1
        loss_attn_target = cross_entropy(attn_map[:, 0, :], target_attn_gt)
        
        # Attention Supervision for Target Pos
        # target_pos_attn = torch.gather(attn_map2[:, 0, :], 1, attn_index)
        # loss_target_pos_attn = criterion(target_pos_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device))
        attn_index = torch.tensor(attn_index, dtype=torch.int64).to(device)
        target_pos_attn_gt = F.one_hot(attn_index, num_classes=attn_map.shape[2])
        loss_target_pos_attn = cross_entropy(attn_map2[:, 0, :], target_pos_attn_gt)

        # Attention Loss
        loss_attn = (loss_attn_target + loss_target_pos_attn) * 5000

        if stage > 0:
            # Attention Supervision for Target Pos, EEF Pos, Command
            # traj_attn = attn_map3[:, 1, 0] + attn_map3[:, 1, -2] + attn_map3[:, 1, -1]
            # loss_traj_attn = criterion(traj_attn, torch.ones(attn_map3.shape[0], 1, dtype=torch.float32).to(device))
            traj_attn_gt = torch.zeros(attn_map.shape[0], attn_map.shape[2], dtype=torch.float32).to(device)
            traj_attn_gt[:, 0] = 1
            traj_attn_gt[:, -2] = 1
            traj_attn_gt[:, -1] = 1
            loss_traj_attn = cross_entropy(attn_map3[:, 1, :], traj_attn_gt)
            loss_attn = loss_attn + loss_traj_attn * 5000

            # Only training on xyz, ignoring rpy
            # For trajectory, use a pre-defined weight matrix to indicate the importance of the trajectory points
            trajectory_pred = trajectory_pred * mask
            ee_traj = ee_traj * mask
            weight_matrix = torch.tensor(np.array([0.9 ** i for i in range(ee_traj.shape[-1])]), dtype=torch.float32) + torch.tensor(np.array([0.9 ** i for i in range(ee_traj.shape[-1]-1, -1, -1)]), dtype=torch.float32)
            # weight_matrix = weight_matrix
            weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(1).repeat(ee_traj.shape[0], ee_traj.shape[1], 1).cuda()
            loss1 = (criterion2(trajectory_pred, ee_traj) * weight_matrix).sum() / (mask * weight_matrix).sum()
            writer.add_scalar('train loss traj', loss1.item(), global_step=epoch_idx * len(data_loader) + idx)
            loss = loss5 + loss1
            print('loss1', loss1.item())

        loss = loss + loss_attn


        print(f'{loss_attn_target.item() * 5000:.2f}')
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
        print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l5 {loss5.item():.2f}')

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


def test(writer, name, epoch_idx, data_loader, model, criterion, train_dataset_size, stage, print_attention_map=False):
    with torch.no_grad():
        model.eval()
        error_trajectory = 0
        error_gripper = 0
        loss5_accu = 0
        idx = 0
        error_target_position = 0
        error_displacement = 0
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
        
        for idx, (img, target, joint_angles, ee_pos, ee_traj, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj) in enumerate(data_loader):
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
                target_position_pred, attn_map, attn_map2 = model(img, ee_pos, sentence, phis, stage)
            else:
                target_position_pred, attn_map, attn_map2, attn_map3, trajectory_pred = model(img, ee_pos, sentence, phis, stage)


            target_pos = target_pos.detach().cpu()
            target_position_pred = target_position_pred.detach().cpu()
            error_target_position_this_time = torch.sum(((target_position_pred[:, :3] - target_pos[:, :3]) * torch.tensor(std[:3])) ** 2, axis=1) ** 0.5
            error_target_position += error_target_position_this_time.sum()
            num_datapoints += error_target_position_this_time.shape[0]

            if stage > 0:
                trajectory_pred = trajectory_pred * mask
                ee_traj = ee_traj * mask
                # Only training on xyz, ignoring rpy
                loss1 = criterion2(trajectory_pred, ee_traj).sum() / mask.sum()

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
                if stage > 0:
                    trajectory_pred = trajectory_pred * std_traj_gripper
                    target_position_pred = target_position_pred * std
                    target_pos = target_pos * std
                    ee_traj = ee_traj * std_traj_gripper
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
                    ax.imshow(img[0].detach().cpu().numpy()[::-1, :, :])


                    ax = fig.add_subplot(1, 3, 3)
                    ax.plot(gripper_x, gripper)
                    ax.plot(gripper_x, gripper_pred)

                    # plt.show()

                    if not os.path.isdir(f'results_png/'):
                        os.mkdir(f'results_png/')
                    if not os.path.isdir(f'results_png/{name}/'):
                        os.mkdir(f'results_png/{name}/')
                    if not os.path.isdir(f'results_png/{name}/{epoch_idx}/'):
                        os.mkdir(f'results_png/{name}/{epoch_idx}/')
                    plt.savefig(os.path.join(f'results_png/{name}/{epoch_idx}/', f'{idx}.png'))

            idx += 1

            # Print
            # print(f'test: epoch {epoch_idx}, step {idx}, loss5 {loss5.item():.2f}')
            if stage == 0:
                print(error_target_position / num_datapoints)
            else:
                print(error_target_position / num_datapoints, error_trajectory / num_trajpoints, error_gripper / num_grippoints)

        # Log
        writer.add_scalar('test error_target_position', error_target_position / num_datapoints, global_step=epoch_idx * train_dataset_size)
        if stage > 0:
            writer.add_scalar('test error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)
            writer.add_scalar('test error_gripper', error_gripper / num_grippoints, global_step=epoch_idx * train_dataset_size)


def main(writer, name, batch_size=96):
    # data_root_path = r'/data/Documents/yzhou298'
    data_root_path = r'/mnt/disk1'
    ckpt_path = os.path.join(data_root_path, r'ckpts/')
    save_ckpt = True
    supervised_attn = True
    curriculum_learning = True
    ckpt = None

    # load model
    model = Backbone(img_size=224, embedding_size=192, num_traces_in=9, num_traces_out=10, num_weight_points=12)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt), strict=True)

    model = model.to(device)

    # load data
    data_dirs = [
        os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224/'),
    ]
    dataset_train = DMPDatasetEERandTarXYLang(data_dirs, random=True, length_total=120)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    dataset_test = DMPDatasetEERandTarXYLang([os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224_test/')], random=True, length_total=36)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    dataset_train_dmp = DMPDatasetEERandTarXYLang(data_dirs, random=False, length_total=120)
    data_loader_train_dmp = torch.utils.data.DataLoader(dataset_train_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=2,
                                          collate_fn=pad_collate_xy_lang)
    dataset_test_dmp = DMPDatasetEERandTarXYLang([os.path.join(data_root_path, 'dataset/mujoco_dataset_pick_push_RGBD_different_angles_224_test/')], random=False, length_total=36)
    data_loader_test_dmp = torch.utils.data.DataLoader(dataset_test_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=2,
                                          collate_fn=pad_collate_xy_lang)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    print('loaded')

    # train n epoches
    loss_stage = 0
    for i in range(0, 500):
        if loss_stage == 0:
            loss_stage = train(writer, name, i, data_loader_train, model, optimizer, scheduler,
                criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=False)
            test(writer, name, i + 1, data_loader_test, model, criterion, len(data_loader_train), loss_stage, print_attention_map=True)
        else:
            loss_stage = train(writer, name, i, data_loader_train_dmp, model, optimizer, scheduler,
                criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=False)
            test(writer, name, i + 1, data_loader_test_dmp, model, criterion, len(data_loader_train_dmp), loss_stage, print_attention_map=True)
        if i >= 5:
            loss_stage = 1



if __name__ == '__main__':
    name = 'train-11-rgbd-crossentropy-mse-lr-1e-4'
    writer = SummaryWriter('runs/' + name)
    main(writer, name)
