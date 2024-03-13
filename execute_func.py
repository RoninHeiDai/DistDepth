# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import imageio
import json
import kornia
import numpy as np
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
torch.backends.cudnn.benchmark = True

from utils import *
from layers import *
import datasets
import networks
from dpt_networks.dpt_depth import DPTDepthModel, DPTDepthModel2


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        # 由frame ids的长度来判断输入的帧数
        self.num_input_frames = len(self.opt.frame_ids)
        # 位姿个数与帧数相同
        self.num_pose_frames = self.num_input_frames
        '''
            这里的frame_ids实际上是一些参数，例如0代表当前输入的图片，如果是[0,-1,1]，则是代表当前帧，前一帧和后一帧。
            后面加入的“s”参数则表示采用stereo training（图像对训练）。
            opt参数主要由训练时终端输入
        '''
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        # 只有当使用当前帧以及图像对时不需要posenet
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        # 使用resnet作为基础模型encoder，depth decoder作为decoder
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        # 加载专家训练网络DPT。！！！！此处需要修改，加载不同的模型需要修改不同模型的代码
        # Download pretrained weights from DPT (https://github.com/isl-org/DPT) and put them under './weights/'
        self.mono_model = DPTDepthModel(
            #path='./weights/dpt_hybrid-midas-501f0c75.pt', # general purpose
            path='./weights/dpt_hybrid_nyu-2ce69ec7.pt',  # indoor
            #path='./weights/dpt_large-midas-2f21e586.pt', # general purpose
            backbone="vitb_rn50_384", #DPT-hybrid (default)
            #backbone="vitl16_384", # DPT-Large (use with dpt-large)
            non_negative=True,
        )

        ### use NYU-finetuned weights, note that this model's output is in depth space, so no need to invert again in L252
        # self.mono_model = DPTDepthModel2(
        #     path='./weights/dpt_hybrid_nyu-2ce69ec7.pt',
        #     scale=0.000305,
        #     shift=0.1378,
        #     invert=True,
        #     backbone="vitb_rn50_384",
        #     non_negative=True,
        # )
        self.mono_model.requires_grad=False
        self.mono_model.to(self.device)
        # pose估计网络默认为resnet50。！！！！此处看情况，是否需要更换不同的pose net。更换则需要更改
        # By default, we use a standalone ResNet50 as PoseNet.
        if self.use_pose_net:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                50, # revise this number if you use a different ResNet backbone
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
        # 网络基本参数。！！！！此处需要更改，根据不同的网络更改训练参数
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate) #optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        # ！！！！此处的数据集dict需要增加我们本身的数据集
        # data
        datasets_dict = {
                         "SimSIN": datasets.SimSINDataset,
                         "VA": datasets.VADataset,
                         "NYUv2": datasets.NYUv2Dataset,
                         "UniSIN": datasets.UniSINDataset,}
        self.dataset = datasets_dict[self.opt.dataset]
        # 该参数为数据集的flag参数。！！！！此处也可能需要加入自身数据集
        #self.approx_factor = 1.0
        # set to 1.0: using default SimSIN. VA's alignment is approximately 2x for depth trained on SimSIN
        if self.opt.dataset == 'SimSIN':
            self.approx_factor = 1.0
        elif self.opt.dataset == 'VA':
            self.approx_factor = 2.0
        else:
            self.approx_factor = 1.0


        fpath = os.path.join(self.opt.data_path,  "{}.txt")

        # The below is sample code for training on VA and Replica
        # 包括训练集和验证集
        # ！！！！此处的数据集选项需要增加本身数据集选项
        if self.opt.dataset == 'VA':
            train_filenames = readlines(fpath.format("VA_all"))
            val_filenames = readlines(fpath.format("VA_left_all"))
        elif self.opt.dataset == 'SimSIN':
            train_filenames = readlines(fpath.format("replica_train"))
            val_filenames = readlines(fpath.format("replica_test_sub"))
        else:
            raise NotImplementedError("Please define your training and validation file path")
        # define train/val file list for SimSIN or UniSIN in the under. DOWNLOAD the data in the project page
        # train_filenames = readlines(fpath.format("all_large_release2")) # readlines(fpath.format("UniSIN_500_list"))
        # val_filenames = readlines(fpath.format("replica_test_sub")
        # 训练总的步数，epoch是整个数据集的轮数
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        # 训练初始化操作，包括数据集，数据集的训练参数；以及验证集
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        # ！！！开始的huberloss等等为什么，不是很理解
        self.ssim = SSIM()
        self.ssim.to(self.device)
        self.depth_criterion = nn.HuberLoss(delta=0.8)
        self.SOFT = nn.Softsign()
        self.ABSSIGN = torch.sign
        # 将深度图转化成point cloud
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_mn", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        try:
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))
        except:
            print("In reference mode! There are {:d} samples\n".format(len(val_dataset)))

        self.save_opts()
        self.cnt = -1
        self.ones = torch.ones(self.opt.batch_size,1,256,256,1).cuda()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        # 训练每个epoch，每当训练次数达到frequency时保存model
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            self.mode = 'train'
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                # ！！！这里的depth_gt没有理解，这里的inputs应该是上面for循环中train loader的参数，但是没找到具体指代
                # 应该是数据集的inputs之类的，数据集的py文件有该变量相关
                if "depth_gt" in inputs:
                    self.compute_depth_losses_Hab(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """
        Pass a minibatch through the network and generate images and losses
        """
        # input images are in [0,1]
        # 加载数据到gpu

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # 通过depth decoder得到视差图，深度图应该是下面generate_images_pred函数生成
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        outputs = self.models["depth"](features)

        # Monocular depth cues. It needs to normalize from [0,1] to [-1,-1] to accomodate DPT input.
        # [Optional] You can comment out all outputs["fromMono_disparity"] at test time to speed up
        # The output range of outputs["fromMono_disparity"] is large, approx 0-2000
        # 此处应该为使用DPT专家网络得到更准确的depth。理论上可能需要修改！！！！
        outputs["fromMono_disparity"], feature_dpt = self.mono_model(inputs[("color_aug", 0, 0)])

        # 600 is a stablization term for SSIM loss calculation (statistical distillation loss)
        # Outputs["fromMono"] is in disparity space. +1.0 is to avoid divide by zero.
        outputs["fromMono_depth"] = 1/(outputs["fromMono_disparity"]+1.0)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs))
        # 生成深度图
        self.generate_images_pred(inputs, outputs)
        # 计算误差loss
        if self.mode == 'train':
            losses = self.compute_losses(inputs, outputs)
        elif self.mode == 'val':
            losses={}

        return outputs, losses

    def val(self):
        """
        Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.__next__()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.__next__()

        with torch.no_grad():
            # 计算loss 的函数process_batch，在上面已经定义
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses_Hab(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, outputs, feats=None):
        """
        Combining monodepth2 and distillation losses
        计算loss的函数，在process_batch中使用过
        loss_spatial_dist与loss_stat_dist为结构蒸馏的空间损失与统计损失
        Combining monodepth2 and distillation losses
        """
        losses = {}
        stereo_loss = 0
        # ！！！该for循环关于stereo loss的计算没有看懂，reprojection loss以及identity reprojection loss的问题。
        # 个人认为此处应该是根据monodepth2的loss计算然后加上知识蒸馏部分的loss
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            # only use souce scale for loss
            source_scale = 0

            disp = outputs[("out", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            # 此处重点理解！这里的frame ids实际上包括左右一致性和光度一致性的loss，因为如果包括stereo train的话
            # 那么frame ids中会有's'这个元素，也就是相当于计算了左右一致性的loss。其他的元素如-1，1则是计算光度一致性的loss！

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            # 此处是拼接上面的各种reprojection loss
            reprojection_losses = torch.cat(reprojection_losses, 1)

            # auto-masking
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss = identity_reprojection_losses
            # save both images, and do min all at once below
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            if combined.shape[1] == 1:
                to_optimize = combined
            else:
                to_optimize, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimize.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # 这里的smooth loss为视差图像的平滑性损失
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            stereo_loss += loss
            losses["loss/{}".format(scale)] = loss

        stereo_loss /= self.num_scales

        losses["loss"] = stereo_loss

        # Deprecated!
        # median alignment from fromMono_depth to out depth
        # ease to use distillation
        # outputs["fromMono_depth"] = (600.0/(outputs["fromMono_disparity"]+1.0))
        # fac = (torch.median(outputs[('depth', 0, 0)]) / torch.median(outputs["fromMono_depth"])).detach()
        # target_depth = outputs["fromMono_depth"]*fac

        # linear alignment from fromMono_depth to out depth
        # solve least square for alignment
        A = torch.cat((outputs["fromMono_depth"][:,:,:,:,None], self.ones), dim=4)
        B = torch.cat((outputs[('depth', 0, 0)][:,:,:,:,None], self.ones), dim=4)
        X = torch.linalg.lstsq(A, B).solution
        a_s = torch.nanmean(X[:,:,:,0,0]).detach()
        a_t = torch.nanmean(X[:,:,:,1,0]).detach()
        target_depth = outputs["fromMono_depth"]* a_s + a_t

        # spatial gradient
        edge_target = kornia.filters.spatial_gradient(target_depth)
        edge_pred = kornia.filters.spatial_gradient(outputs[('depth', 0, 0)])

        # convert to magnitude map
        edge_target =  torch.sqrt(edge_target[:,:,0,:,:]**2 + edge_target[:,:,1,:,:]**2 + 1e-6)
        edge_target = F.normalize(edge_target.view(edge_target.size(0), -1), dim=1, p=2).view(edge_target.size())
        edge_target = edge_target[:,:,5:-5,5:-5]

        # thresholding
        bar_target = torch.quantile(edge_target.reshape(edge_target.size(0), -1), self.opt.thre, dim=1)
        bar_target = bar_target[:, None, None, None]
        pos = edge_target > bar_target
        mask_target = self.ABSSIGN(edge_target - bar_target)[pos]
        mask_target = mask_target.detach()

        # convert prediction to magnitude map
        edge_pred =  torch.sqrt(edge_pred[:,:,0,:,:]**2 + edge_pred[:,:,1,:,:]**2 + 1e-6)
        edge_pred = F.normalize(edge_pred.view(edge_pred.size(0), -1), dim=1, p=2).view(edge_pred.size())
        edge_pred = edge_pred[:,:,5:-5,5:-5]
        bar_pred = torch.quantile(edge_pred, self.opt.thre).detach()

        # soft sign for differentiable
        mask_pred = self.SOFT(edge_pred - bar_pred)[pos]

        loss_spatial_dist = 0.001 * self.depth_criterion(mask_pred, mask_target)
        loss_stat_dist = self.compute_ssim_loss(target_depth, outputs[('depth', 0, 0)]).mean()

        losses["loss/pseudo_depth"] = loss_stat_dist + loss_spatial_dist
        losses["loss"] += self.opt.dist_wt * losses["loss/pseudo_depth"]

        #self.cnt += 1
        #print(f'Iter {self.cnt}: {losses["loss"]}')

        return losses

    def compute_reprojection_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images

        计算reprojection_loss在compute_loss中使用
        Computes reprojection loss between a batch of predicted and target images

        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_ssim_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        """
        return self.ssim(pred, target).mean(1, True)

    def compute_depth_losses_Hab(self, inputs, outputs, losses):
        """
        Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [512, 512], mode="bilinear", align_corners=False), 1e-3, 10)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = torch.logical_and(depth_gt>0.01, depth_gt<=10.0)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        主要用于重新投影生成彩色图像(wrapping操作)
        """
        for scale in self.opt.scales:
            disp = outputs[("out", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            depth = output_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # auto-masking
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def eval_save(self):
        """
        save prediction for a minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.__next__()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.__next__()

        with torch.no_grad():
            inputs[("color_aug", 0, 0)] = inputs[("color_aug", 0, 0)].cuda()
            features = self.models["encoder"](inputs[("color_aug", 0, 0)]) #
            outputs = self.models["depth"](features)
            depth = output_to_depth(outputs[('out', 0)], self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, 0)] = depth

            sz = (640,640)
            store_path = 'results/VA'
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            img = inputs[('color_aug',0, 0)]
            img = F.interpolate(img, sz, mode='bilinear', align_corners=True)

            img = img.cpu().numpy().squeeze()
            if img.shape[0] == self.opt.batch_size: # batch_size > 1
                img = np.transpose(img, (0,2,3,1))
            else: # batch_size == 1
                img = np.transpose(img, (1,2,0))

            depth = outputs[('depth', 0, 0)] * self.approx_factor #approximate alignment for visualization
            depth = F.interpolate(depth, sz, mode='bilinear', align_corners=True)
            depth = depth.cpu().numpy().squeeze()

            bsz = img.shape[0]

            if bsz != self.opt.batch_size: # save only one image
                imageio.imwrite(f'{store_path}/00_current.png', img)
                write_turbo_depth_metric(f'{store_path}/00_depth.png', depth, vmax=10.0)
            else: # forloop to save images
                for idx in range(bsz):
                    imageio.imwrite(f'{store_path}/{idx:02d}_current.png', img[idx])
                    write_turbo_depth_metric(f'{store_path}/{idx:02d}_depth.png', depth[idx], vmax=10.0)

            del inputs, outputs

    def eval_save_all(self):
        """
        save prediction for all data on the list
        """
        self.set_eval()
        self.count = 0
        while True:
            try:
                inputs = self.val_iter.__next__()
            except StopIteration:
                break
            # 利用cuda预测出原图的深度图并保存
            with torch.no_grad():
                inputs[("color_aug", 0, 0)] = inputs[("color_aug", 0, 0)].cuda()
                features = self.models["encoder"](inputs[("color_aug", 0, 0)]) #
                outputs = self.models["depth"](features)
                depth = output_to_depth(outputs[('out', 0)], self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, 0)] = depth
                sz = (640,640)
                store_path = f'results_all/'
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                    os.makedirs(store_path+'/image')
                    os.makedirs(store_path+'/depth')
                # 数组采样。详细看interpolate函数用法：https://blog.csdn.net/qq_50001789/article/details/120297401
                img = inputs[('color',0, 0)]
                img = F.interpolate(img, sz, mode='bilinear', align_corners=True)
                img = img.cpu().numpy().squeeze()

                if img.ndim == 3:
                    img = np.transpose(img, (1,2,0))
                else:
                    raise ValueError('Eval_save_all only supports batch_size = 1')

                if 'depth_gt' in inputs:
                    depth_gt = inputs['depth_gt']
                    depth_gt = F.interpolate(depth_gt, sz, mode='bilinear', align_corners=True)
                    depth_gt = depth_gt.cpu().numpy().squeeze()

                depth = outputs[('depth', 0, 0)] * self.approx_factor #approximate alignment for visualization
                depth = F.interpolate(depth, sz, mode='bilinear', align_corners=True)
                depth = depth.cpu().numpy().squeeze()

                self.count += 1
                imageio.imwrite(f'{store_path}/image/{self.count:04d}_img.png', img)
                write_turbo_depth_metric(f'{store_path}/depth/{self.count:04d}_depth.png', depth, vmax=10.0)
                if 'depth_gt' in inputs:
                    write_turbo_depth_metric(f'{store_path}/depth/{self.count:04}_depth_gt.png', depth_gt, vmax=10.0)

            del inputs, outputs

    def eval_measure(self):
        """
        eval on either VA or NYUv2
        """
        self.set_eval()
        self.abs_mn = AverageMeter('abs_mean')
        self.abs_rel = AverageMeter('abs_rel')
        self.sq_rel = AverageMeter('sq_rel')
        self.rms = AverageMeter('rms')
        self.log_rms = AverageMeter('log_rms')
        self.a1 = AverageMeter('a1')
        self.a2 = AverageMeter('a2')
        self.a3 = AverageMeter('a3')
        self.metr = [self.abs_mn, self.abs_rel, self.sq_rel, self.rms, self.log_rms, self.a1, self.a2, self.a3]
        N = self.opt.batch_size

        local_count = 0
        losses = {}
        while True:
            try:
                inputs = self.val_iter.__next__()
            except StopIteration:
                if not local_count == 0:
                    break
                else:
                    self.val_iter = iter(self.val_loader)
                    inputs = self.val_iter.__next__()

            with torch.no_grad():
                inputs[("color_aug", 0, 0)] = inputs[("color_aug", 0, 0)].cuda()
                inputs["depth_gt"] = inputs["depth_gt"].cuda()
                features = self.models["encoder"](inputs[("color_aug", 0, 0)])
                outputs = self.models["depth"](features)
                depth = output_to_depth(outputs[('out', 0)], self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, 0)] = depth
                if "depth_gt" in inputs:
                    if self.opt.dataset == 'VA':
                        self.compute_depth_errors_VA(inputs, outputs, losses)
                    elif self.opt.dataset == 'NYUv2':
                        self.compute_depth_errors_NYUv2(inputs, outputs, losses)
                    else:
                        raise NotImplementedError("Do evaluation only on VA or NYUv2")

                    for var, name in zip(self.metr, self.depth_metric_names):
                        var.update(losses[name], N)

            local_count += 1

        if "depth_gt" in inputs:
            idfy = self.opt.load_weights_folder
            f = open(f'evaluation-{idfy}.txt','w')
            for var in self.metr:
                f.write(str(var))
            f.close()

        del inputs, outputs, losses

    def compute_depth_errors_VA(self, inputs, outputs, losses):
        """
        compute depth errors on VA
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(depth_pred, [640, 640], mode="bilinear", align_corners=True)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10.0)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        if losses is None:
            losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # 这里的num_pose_frames实际上是opt frame ids的长度，理论上来说这个ids的长度只有三种情况
        # [0,-1,1]依靠前后帧进行训练
        # [0,'s']依靠双目图像对进行训练
        # [0,-1,1,'s']依靠前后帧以及双目图像对进行训练
        # 此处的num_pose_frames为位姿帧的个数，与输入帧的个数相关
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    # 总是保持按时间顺序传递帧
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    # 通过pose解码器得到输出的 轴角 和 平移
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative   帧id为负的话则反转矩阵
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            # 将所有帧一起输入到姿态网络（并预测所有姿态）

            pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def compute_depth_errors_NYUv2(self, inputs, outputs, losses):
        """
        compute depth errors on NYUv2
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(depth_pred, [448, 608], mode="bilinear", align_corners=True)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10.0)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        if losses is None:
            losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """
        print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """
        write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "inverse_depth_output_{}/{}".format(s, j),
                    normalize_image(outputs[("out", s)][j]), self.step)

                # automasking
                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def log_losses(self, mode, losses):
        """
        write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """
        save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """
        save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

    def load_model(self):
        """
        load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def eval_measure_multi(self):
        self.dataset = datasets.VADataset
        fpath = os.path.join(self.opt.data_path,  "{}.txt")
        val_filenames = readlines(fpath.format("UE4_left_freq_5"))
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=8, pin_memory=False, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["encoder"] = networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        self.models["mono_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained")
        self.models["mono_encoder"].to(self.device)

        self.models["mono_depth"] = \
            networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales)
        self.models["mono_depth"].to(self.device)

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        if self.opt.load_weights_folder is not None:
            self.load_model_multi()


        self.set_eval()

        self.abs_mn = AverageMeter('abs_mean')
        self.abs_rel = AverageMeter('abs_rel')
        self.sq_rel = AverageMeter('sq_rel')
        self.rms = AverageMeter('rms')
        self.log_rms = AverageMeter('log_rms')
        self.a1 = AverageMeter('a1')
        self.a2 = AverageMeter('a2')
        self.a3 = AverageMeter('a3')
        self.metr = [self.abs_mn, self.abs_rel, self.sq_rel, self.rms, self.log_rms, self.a1, self.a2, self.a3]
        N = self.opt.batch_size

        #count = 0
        while True:
            try:
                inputs = self.val_iter.__next__()
            except StopIteration:
                break

            with torch.no_grad():
                outputs, losses = self.process_batch_multi(inputs)

                if "depth_gt" in inputs:
                    self.self.compute_depth_errors_VA(inputs, outputs, losses)

                    for var, name in zip(self.metr, self.depth_metric_names):
                        var.update(losses[name], N)

        if "depth_gt" in inputs:
            idfy = self.opt.load_weights_folder
            f = open(f'evaluation-{idfy}.txt','w')
            for var in self.metr:
                f.write(str(var))
            f.close()

        del inputs, outputs, losses

    def predict_poses_multi(self, inputs):
        """
        Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    def process_batch_multi(self, inputs, is_train=False):
        """
        Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        with torch.no_grad():
            pose_pred = self.predict_poses(inputs)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # single frame path

        with torch.no_grad():
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models['mono_depth'](feats))

        self.generate_images_pred(inputs, mono_outputs)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, _, _ = self.models["encoder"](inputs["color_aug", 0, 0],
                            lookup_frames,
                            relative_poses,
                            inputs[('K', 2)],
                            inputs[('inv_K', 2)],
                            min_depth_bin=min_depth_bin,
                            max_depth_bin=max_depth_bin)
        outputs.update(self.models["depth"](features))

        self.generate_images_pred_multi(inputs, outputs)
        losses = {}

        return outputs, losses

    def generate_images_pred_multi(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("out", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            depth = output_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

    def load_mono_model(self):
        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model_multi(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
