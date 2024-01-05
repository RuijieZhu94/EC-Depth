from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
import copy

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.student_models = {}
        self.student_parameters_to_train = []

        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        if self.opt.ddp:
            dist.init_process_group(backend='nccl', )
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        self.student_models["encoder"] = networks.mpvit_small()
        self.student_models["encoder"].num_ch_enc = [64,128,216,288,288]
        if self.opt.ddp:
            self.student_models["encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_models["encoder"])
        self.student_models["encoder"].to(self.device)
 
        self.student_models["depth"] = networks.DepthDecoder(
            self.student_models["encoder"].num_ch_enc, self.opt.scales)
        if self.opt.ddp:
            self.student_models["depth"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_models["depth"])
        self.student_models["depth"].to(self.device)
        self.student_parameters_to_train += list(self.student_models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.student_models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                if self.opt.ddp:
                    self.student_models["pose_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_models["pose_encoder"])
                self.student_models["pose_encoder"].to(self.device)
                self.student_parameters_to_train += list(self.student_models["pose_encoder"].parameters())

                self.student_models["pose"] = networks.PoseDecoder(
                    self.student_models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.student_models["pose"] = networks.PoseDecoder(
                    self.student_models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.student_models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
            if self.opt.ddp:
                self.student_models["pose"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_models["pose"])
            self.student_models["pose"].to(self.device)
            self.student_parameters_to_train += list(self.student_models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.student_models["predictive_mask"] = networks.DepthDecoder(
                self.student_models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.student_models["predictive_mask"].to(self.device)
            self.student_parameters_to_train += list(self.student_models["predictive_mask"].parameters())

        
        
        self.teacher_models = copy.deepcopy(self.student_models)

        self.student_params = [{
                "params":self.student_parameters_to_train, 
                "lr": 1e-4
                #"weight_decay": 0.01
            },
            {
                "params": list(self.student_models["encoder"].parameters()), 
                "lr": self.opt.learning_rate
                #"weight_decay": 0.01
            }]
        self.student_model_optimizer = optim.AdamW(self.student_params)
        self.student_model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.student_model_optimizer,0.9)

        if self.opt.load_weights_folder is not None:
            self.load_teacher_model()
            self.load_student_model()

        if self.opt.ddp:
            for key in self.student_models.keys():
                self.student_models[key] = DDP(self.student_models[key], device_ids=[self.local_rank],
                     output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True)
                self.teacher_models[key] = DDP(self.teacher_models[key], device_ids=[self.local_rank],
                     output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True)

        if self.local_rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
            print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        # num_train_samples = len(train_filenames)
        # self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, augmix=(self.opt.augmix or self.opt.aug_fp))
        if self.opt.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        else:        
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker)

        self.num_total_steps = len(self.train_loader) * self.opt.num_epochs

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        if self.opt.ddp:
            rank, world_size = get_dist_info()
            self.world_size = world_size
            val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False, sampler=val_sampler)    
        else: 
            self.world_size = 1    
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            if self.local_rank == 0:            
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if self.local_rank == 0:
            print("Using split:\n  ", self.opt.split)
            # print("There are {:d} training items and {:d} validation items\n".format(
            #     len(train_dataset), len(val_dataset)))
        if self.opt.ddp:
            self.opt.log_frequency = self.opt.log_frequency//self.world_size
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.teacher_models.values():
            m.train()
        for m in self.student_models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.teacher_models.values():
            m.eval()
        for m in self.student_models.values():
            m.eval()

    def update_ema(self):
        # for key, value in 
        # self.teacher_parameters_to_train = self.opt.ema_weight * self.teacher_parameters_to_train \
        #     + (1-self.opt.ema_weight) * self.student_parameters_to_train
        with torch.no_grad():
            # msd = self.student_models.module.state_dict() if self.opt.ddp else self.student_models.state_dict()
            for n in self.opt.models_to_load:
                # if n == "depth" or n == "encoder":
                msd = self.student_models[n].state_dict()
                for k, v in self.teacher_models[n].state_dict().items():
                    if v.dtype.is_floating_point:
                        v *= self.opt.ema_weight
                        v += (1. - self.opt.ema_weight) * msd[k].detach()
                        # 
                        self.teacher_models[n].state_dict().update({k:v})
            print_string = "update teacher model, step: {:>6}, epoch:{:>3}"
            if self.local_rank == 0:
                print(print_string.format(self.step, self.epoch))


    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.opt.ddp:
                self.train_loader.sampler.set_epoch(self.epoch)

            self.run_epoch()

            # begin epoch setting
            if self.epoch > 5:
                self.update_ema()
            
            ### remember to change according to the epoch
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.epoch>0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        if self.local_rank == 0:
            print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            teacher_outputs, teacher_outputs_aug1, teacher_outputs_aug2, outputs, student_outputs_aug1, student_outputs_aug2, losses = self.process_batch(inputs)

            self.student_model_optimizer.zero_grad()
            losses["loss"].backward()
            self.student_model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0 and self.step > 2000

            if early_phase or late_phase:
                if self.local_rank == 0:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                if self.local_rank == 0:
                    self.log("train", inputs, teacher_outputs, teacher_outputs_aug1, teacher_outputs_aug2, outputs, student_outputs_aug1, student_outputs_aug2, losses)
                # self.val()

            self.step += 1
        self.student_model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        teacher_outputs = {}
        teacher_outputs_aug1 = {}
        teacher_outputs_aug2 = {}
        student_outputs = {}
        student_outputs_aug1 = {}
        student_outputs_aug2 = {}
        student_outputs_fp = {}
        # teacher
        with torch.no_grad():
            teacher_images_all = torch.cat((inputs["color", 0, 0],
                                    inputs["color_aug1", 0, 0],
                                    inputs["color_aug2", 0, 0]), 0)
            teacher_features_all = self.teacher_models["encoder"](teacher_images_all)
            teacher_outputs_all = self.teacher_models["depth"](teacher_features_all)
            for key, value in teacher_outputs_all.items():
                teacher_outputs[key], teacher_outputs_aug1[key], teacher_outputs_aug2[key] = torch.split(value, inputs["color", 0, 0].size(0))
        if self.opt.augmix:
            # student
            images_all = torch.cat((inputs["color", 0, 0],
                                    inputs["color_aug1", 0, 0],
                                    inputs["color_aug2", 0, 0]), 0)
            student_features_all = self.student_models["encoder"](images_all)
            student_outputs_all = self.student_models["depth"](student_features_all)
            for key, value in student_outputs_all.items():
                student_outputs[key], student_outputs_aug1[key], student_outputs_aug2[key] = torch.split(value, inputs["color", 0, 0].size(0))

        elif self.opt.fp:
            # student
            student_features = self.student_models["encoder"](inputs["color", 0, 0])
            student_features_fp = [nn.Dropout2d(0.5)(feature) for feature in student_features]
            student_outputs = self.student_models["depth"](student_features)
            student_outputs_fp = self.student_models["depth"](student_features_fp)

        elif self.opt.aug_fp:
            # student
            images_all = torch.cat((inputs["color", 0, 0], 
                                inputs["color_aug1", 0, 0],
                                inputs["color_aug2", 0, 0]), 0)
            student_features_all = self.student_models["encoder"](images_all)
            student_features = [torch.split(student_feature, inputs["color", 0, 0].size(0))[0] for student_feature in student_features_all]
            student_features_fp = [nn.Dropout2d(0.5)(feature) for feature in student_features]
            for i in range(len(student_features_all)):
                student_features_all[i] = torch.cat((student_features_all[i], student_features_fp[i]), dim=0)
            student_outputs_all = self.student_models["depth"](student_features_all)
            for key, value in student_outputs_all.items():
                student_outputs[key], student_outputs_aug1[key], student_outputs_aug2[key], student_outputs_fp[key] = torch.split(value, inputs["color", 0, 0].size(0))

        if self.use_pose_net:
            student_pose = self.student_predict_poses(inputs)
            student_outputs.update(student_pose)
            # student_outputs_aug1.update(student_pose)
            # student_outputs_aug2.update(student_pose)
            # student_outputs_fp.update(student_pose)
            teacher_pose = self.teacher_predict_poses(inputs)
            teacher_outputs.update(teacher_pose)
            teacher_outputs_aug1.update(teacher_pose)
            teacher_outputs_aug2.update(teacher_pose)



        self.generate_images_pred(inputs, student_outputs, reprojection=True)
        self.generate_images_pred(inputs, teacher_outputs, reprojection=True)
        self.generate_images_pred(inputs, teacher_outputs_aug1, reprojection=False)
        self.generate_images_pred(inputs, teacher_outputs_aug2, reprojection=False)
        if self.opt.augmix or self.opt.aug_fp:
            self.generate_images_pred(inputs, student_outputs_aug1)
            self.generate_images_pred(inputs, student_outputs_aug2)
        if self.opt.fp or self.opt.aug_fp:
            self.generate_images_pred(inputs, student_outputs_fp)
        losses = self.compute_student_losses(inputs, teacher_outputs, teacher_outputs_aug1, teacher_outputs_aug2, student_outputs, student_outputs_aug1, student_outputs_aug2, student_outputs_fp)

        return teacher_outputs, teacher_outputs_aug1, teacher_outputs_aug2, student_outputs, student_outputs_aug1, student_outputs_aug2, losses

    def teacher_predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.teacher_models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.teacher_models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.teacher_models["pose_encoder"](pose_inputs)]

            # elif self.opt.pose_model_type == "shared":
            #     pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.teacher_models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs
    
    def student_predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.student_models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.student_models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.student_models["pose_encoder"](pose_inputs)]

            # elif self.opt.pose_model_type == "shared":
            #     pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.student_models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            teacher_outputs, outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            if self.local_rank == 0:
                self.log("val", inputs, teacher_outputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, reprojection=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            if reprojection:
                for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)]

                    # from the authors of https://arxiv.org/abs/1712.00175
                    if self.opt.pose_model_type == "posecnn":

                        axisangle = outputs[("axisangle", 0, frame_id)]
                        translation = outputs[("translation", 0, frame_id)]

                        inv_depth = 1 / depth
                        mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                        T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

                    outputs[("sample", frame_id, scale)] = pix_coords

                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)

                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss # B x 1 x H x W

    def compute_losses(self, inputs, outputs, outputs_aug=None, outputs_aug1=None, outputs_fp=None):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            # augmix
            if self.opt.augmix or self.opt.aug_fp:
                disp_aug = outputs_aug[("disp", scale)]
                disp_aug1 = outputs_aug1[("disp", scale)]
                disp_mix = torch.clamp((disp + disp_aug + disp_aug1) / 3., 1e-7, 1).log()
                loss += 0.005 * (F.kl_div(disp_mix, disp, reduction='batchmean') 
                        + F.kl_div(disp_mix, disp_aug, reduction='batchmean')
                        + F.kl_div(disp_mix, disp_aug1, reduction='batchmean')) / 3.

            # fp
            if self.opt.fp or self.opt.aug_fp:
                disp_fg = outputs_fp[("disp", scale)]
                disp_fg_avg = torch.clamp((disp_fg + disp) / 2, 1e-7, 1).log()
                loss += 0.005* (F.kl_div(disp_fg_avg, disp, reduction='batchmean')
                               + F.kl_div(disp_fg_avg, disp_fg, reduction='batchmean')) / 2

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_student_losses(self, inputs, teacher_outputs, teacher_outputs_aug1, teacher_outputs_aug2, outputs, outputs_aug=None, outputs_aug1=None, outputs_fp=None):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            reprojection_losses_t = []
            # reprojection_losses_aug = []
            # reprojection_losses_aug1 = []
            # # reprojection_loss_aug = []
            # # reprojection_loss_aug1 = []
            # # reprojection_loss_fp = []

            # reprojection and smoothness loss
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            
            
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                T_pred = teacher_outputs[("color", frame_id, scale)]
                # pred_aug = teacher_outputs_aug1[("color", frame_id, scale)]
                # pred_aug1 = teacher_outputs_aug2[("color", frame_id, scale)]
            #     # pred_fp = outputs_fp[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                reprojection_losses_t.append(self.compute_reprojection_loss(T_pred, target))
                # reprojection_losses_aug.append(self.compute_reprojection_loss(pred_aug, target))
                # reprojection_losses_aug1.append(self.compute_reprojection_loss(pred_aug1, target))
            #     # reprojection_loss_fp.append(self.compute_reprojection_loss(pred_fp, target))

            reprojection_losses = torch.cat(reprojection_losses, 1) # B x num_frames x H x W
            reprojection_losses_t = torch.cat(reprojection_losses_t, 1)
            # reprojection_losses_aug = torch.cat(reprojection_losses_aug, 1)
            # reprojection_losses_aug1 = torch.cat(reprojection_losses_aug1, 1)
            # # # reprojection_loss_fp = torch.cat(reprojection_loss_fp, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses   # B x num_frames x H x W

            # # elif self.opt.predictive_mask:
            # #     # use the predicted mask
            # #     mask = outputs["predictive_mask"]["disp", scale]
            # #     if not self.opt.v1_multiscale:
            # #         mask = F.interpolate(
            # #             mask, [self.opt.height, self.opt.width],
            # #             mode="bilinear", align_corners=False)

            # #     reprojection_losses *= mask

            # #     # add a loss pushing mask to 1 (using nn.BCELoss for stability)
            # #     weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            # #     loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                reprojection_loss_t = reprojection_losses_t.mean(1, keepdim=True)
                # reprojection_loss_aug = reprojection_losses_aug.mean(1, keepdim=True)
                # reprojection_loss_aug1 = reprojection_losses_aug1.mean(1, keepdim=True)
            #     # reprojection_loss_fp = reprojection_loss_fp.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
                reprojection_loss_t = reprojection_losses_t
                # reprojection_loss_aug = reprojection_losses_aug
                # reprojection_loss_aug1 = reprojection_losses_aug1
            #     # reprojection_loss_fp = reprojection_loss_fp

            if not self.opt.disable_automasking:
            #     # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)    # B x 2num_frames x H x W
                combined_t = torch.cat((identity_reprojection_loss, reprojection_loss_t), dim=1)
                # combined_aug = torch.cat((identity_reprojection_loss, reprojection_loss_aug), dim=1)
                # combined_aug1 = torch.cat((identity_reprojection_loss, reprojection_loss_aug1), dim=1)
            #     # combined_fp = torch.cat((identity_reprojection_loss, reprojection_loss_fp), dim=1)
            else:
                combined = reprojection_loss
                combined_t = reprojection_loss_t
                # combined_aug = reprojection_loss_aug
                # combined_aug1 = reprojection_loss_aug1
            #     # combined_fp = reprojection_loss_fp

            if combined_t.shape[1] == 1:
                to_optimise = combined
                to_optimise_t = combined_t
                # to_optimise_aug = combined_aug
                # to_optimise_aug1 = combined_aug1
            #     # to_optimise_fp = combined_fp
            else:
                to_optimise, idxs = torch.min(combined, dim=1)  # B x 1 x H x W
                to_optimise_t, idxs_1 = torch.min(combined_t, dim=1)
                # to_optimise_aug, _ = torch.min(combined_aug, dim=1)
                # to_optimise_aug1, _ = torch.min(combined_aug1, dim=1)
            #     # to_optimise_fp, _ = torch.min(combined_fp, dim=1)
            # to_optimise_t = to_optimise_t.unsqueeze(dim=1)
            # to_optimise_aug = to_optimise_aug.unsqueeze(dim=1)
            # to_optimise_aug1 = to_optimise_aug1.unsqueeze(dim=1)
            # to_optimise_all = torch.cat((to_optimise_t, to_optimise_aug, to_optimise_aug1), dim=1)
            # _, idx = torch.min(to_optimise_all, dim=1)
            # idx = idx.unsqueeze(dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs_1 > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            # # # loss += to_optimise_aug.mean()
            # # # loss += to_optimise_aug1.mean()
            # # # loss += to_optimise_fp.mean()

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            smooth_loss = get_smooth_loss(disp, color)
            if self.opt.augmix or self.opt.aug_fp:
                disp_aug = outputs_aug[("disp", scale)]
                disp_aug1 = outputs_aug1[("disp", scale)]
                smooth_loss += get_smooth_loss(disp_aug, color)
                smooth_loss += get_smooth_loss(disp_aug1, color)
            if self.opt.fp or self.opt.aug_fp:
                disp_fp = outputs_fp[("disp", scale)]
                smooth_loss += get_smooth_loss(disp_fp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            
            # pseudo_label loss
            T_depth = teacher_outputs[("depth", 0, scale)]
            T_depth_aug1 = teacher_outputs_aug1[("depth", 0, scale)]
            T_depth_aug2 = teacher_outputs_aug2[("depth", 0, scale)]
            S_depth = outputs[("depth", 0, scale)]

            # T_dep = torch.cat((T_depth_0, T_depth_aug1, T_depth_aug2), dim=1)
            # T_depth = torch.gather(T_dep, 1, idx)

            weight_mask_depth = (torch.abs(T_depth - T_depth_aug1) < self.opt.depth_thre) * (torch.abs(T_depth - T_depth_aug2) < self.opt.depth_thre) 
            weight_mask_depth = weight_mask_depth.squeeze(1)
            weight_mask_repro = to_optimise_t < self.opt.stable_thre  # B x H x W
            weight_mask = weight_mask_depth * weight_mask_repro
            weight_mask = weight_mask.detach()
            N_pixel = 1.0 + weight_mask.sum()
            outputs["weight_mask"] = weight_mask.float()
            # outputs["weight_mask_repro"] = weight_mask_repro.float()
            # outputs["weight_mask_depth"] = weight_mask_depth.float()

            abs_diff = torch.abs(T_depth - S_depth).squeeze(1) * weight_mask
            l1_loss = abs_diff.sum() / N_pixel       # epoch0 thre0.05: 0.0064
            # l1_loss = torch.abs(T_depth - S_depth).mean()
            # TODO: relative depth diff
            if self.opt.augmix or self.opt.aug_fp:
                S_depth_aug = outputs_aug[("depth", 0, scale)]
                S_depth_aug1 = outputs_aug1[("depth", 0, scale)]
                # l1_loss += torch.abs(T_depth - S_depth_aug).mean()
                # l1_loss += torch.abs(T_depth - S_depth_aug1).mean()
                abs_diff = torch.abs(T_depth - S_depth_aug).squeeze(1) * weight_mask
                l1_loss += abs_diff.sum() / N_pixel       # epoch0 thre0.05: 0.0064
                abs_diff = torch.abs(T_depth - S_depth_aug1).squeeze(1) * weight_mask
                l1_loss += abs_diff.sum() / N_pixel       # epoch0 thre0.05: 0.0064
            if self.opt.fp or self.opt.aug_fp:
                S_depth_fp = outputs_fp[("depth", 0, scale)]
                abs_diff = torch.abs(T_depth - S_depth_fp).squeeze(1) * weight_mask
                l1_loss += abs_diff.sum() / N_pixel       # epoch0 thre0.05: 0.0064
                # l1_loss += torch.abs(T_depth - S_depth_fp).mean()
            

            if self.opt.no_ssim:
                loss += l1_loss
            else:
                ssim_loss = self.ssim(S_depth, T_depth).mean()
                if self.opt.augmix or self.opt.aug_fp:
                    ssim_loss += self.ssim(S_depth_aug, T_depth).mean()
                    ssim_loss += self.ssim(S_depth_aug1, T_depth).mean()
                if self.opt.fp or self.opt.aug_fp:
                    ssim_loss += self.ssim(S_depth_fp, T_depth).mean()
                loss += (0.85 * ssim_loss + 0.15 * l1_loss) * self.opt.pseudo_weight
                
            # augmix
            # if self.opt.augmix or self.opt.aug_fp:
            #     T_disp = teacher_outputs[("disp", scale)]
            #     # disp_aug = outputs_aug[("disp", scale)]
            #     # disp_aug1 = outputs_aug1[("disp", scale)]
            #     # disp_mix_old = torch.clamp((disp + disp_aug + disp_aug1) / 3., 1e-7, 1).log()
            #     disp_mix = torch.clamp((T_disp + disp_aug) / 2., 1e-7, 1).log()
            #     disp_mix1 = torch.clamp((T_disp + disp_aug1) / 2., 1e-7, 1).log()
            #     disp_mix0 = torch.clamp((disp_aug + disp_aug1) / 2., 1e-7, 1).log()

            #     loss += 0.0005 * (F.kl_div(disp_mix, disp_aug, reduction='batchmean') + F.kl_div(disp_mix, T_disp, reduction='batchmean')
            #             + F.kl_div(disp_mix1, disp_aug1, reduction='batchmean') + F.kl_div(disp_mix1, T_disp, reduction='batchmean')
            #             + F.kl_div(disp_mix0, disp_aug, reduction='batchmean') + F.kl_div(disp_mix0, disp_aug1, reduction='batchmean')) / 6.

            #     # loss += 0.0005 * (F.kl_div(disp_mix_old, disp, reduction='batchmean') 
            #     #         + F.kl_div(disp_mix_old, disp_aug, reduction='batchmean')
            #     #         + F.kl_div(disp_mix_old, disp_aug1, reduction='batchmean')) / 3.

            # # fp
            # if self.opt.fp or self.opt.aug_fp:
            #     # disp_fp = outputs_fp[("disp", scale)]
            #     disp_fp_avg = torch.clamp((disp_fp + T_disp) / 2, 1e-7, 1).log()
            #     # disp_fg_avg_old = torch.clamp((disp_fg + T_disp) / 2, 1e-7, 1).log()
            #     loss += 0.0005 * (F.kl_div(disp_fp_avg, disp_fp, reduction='batchmean')
            #                       + F.kl_div(disp_fp_avg, T_disp, reduction='batchmean')) / 2.
            #     # loss += 0.0005* (F.kl_div(disp_fg_avg_old, disp, reduction='batchmean')
            #     #                + F.kl_div(disp_fg_avg_old, disp_fg, reduction='batchmean')) / 2.

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        if self.local_rank == 0:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, teacher_outputs, teacher_outputs_aug1, teacher_outputs_aug2, outputs, student_outputs_aug1, student_outputs_aug2, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                # for frame_id in self.opt.frame_ids:
                #     writer.add_image(
                #         "color_{}_{}/{}".format(frame_id, s, j),
                #         inputs[("color", frame_id, s)][j].data, self.step)
                #     writer.add_image(
                #         "color_aug_{}_{}/{}".format(frame_id, s, j),
                #         inputs[("color_aug", frame_id, s)][j].data, self.step)
                #     if s == 0 and frame_id != 0:
                #         writer.add_image(
                #             "color_pred_{}_{}/{}".format(frame_id, s, j),
                #             outputs[("color", frame_id, s)][j].data, self.step)
                if s == 0:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)

                        # if frame_id != 0:
                        #     # writer.add_image(
                        #     #     "T_color_pred_{}_{}/{}".format(frame_id, s, j),
                        #     #     teacher_outputs[("color", frame_id, s)][j].data, self.step)
                        #     # writer.add_image(
                        #     #     "T_color_pred_{}_{}/{}".format(frame_id, s, j),
                        #     #     teacher_outputs[("color", frame_id, s)][j].data, self.step)
                        # else:
                        if frame_id == 0:
                            writer.add_image(
                                "color_{}_{}/{}".format(frame_id, s, j),
                                inputs[("color", frame_id, s)][j].data, self.step)
                            if self.opt.augmix or self.opt.aug_fp:
                                writer.add_image(
                                    "color_aug1_{}_{}/{}".format(frame_id, s, j),
                                    inputs[("color_aug1", frame_id, s)][j].data, self.step)
                                writer.add_image(
                                    "color_aug2_{}_{}/{}".format(frame_id, s, j),
                                    inputs[("color_aug2", frame_id, s)][j].data, self.step)
                        else:
                            writer.add_image(
                                "S_color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)
                            writer.add_image(
                                "T_color_pred_{}_{}/{}".format(frame_id, s, j),
                                teacher_outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "S_disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                writer.add_image(
                    "S_disp_aug_{}/{}".format(s, j),
                    normalize_image(student_outputs_aug1[("disp", s)][j]), self.step)
                writer.add_image(
                    "S_disp_aug1_{}/{}".format(s, j),
                    normalize_image(student_outputs_aug2[("disp", s)][j]), self.step)
                writer.add_image(
                    "T_disp_{}/{}".format(s, j),
                    normalize_image(teacher_outputs[("disp", s)][j]), self.step)
                writer.add_image(
                    "T_disp_aug_{}/{}".format(s, j),
                    normalize_image(teacher_outputs_aug1[("disp", s)][j]), self.step)
                writer.add_image(
                    "T_disp_aug1_{}/{}".format(s, j),
                    normalize_image(teacher_outputs_aug2[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                # elif not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
            writer.add_image(
                "TS_mask/{}".format(j),
                outputs[("weight_mask")][j].data, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        if self.local_rank == 0:
            teacher_save_folder = os.path.join(self.log_path, "models", "teacher", "weights_{}".format(self.epoch))
            if not os.path.exists(teacher_save_folder):
                os.makedirs(teacher_save_folder)

            for model_name, model in self.teacher_models.items():
                save_path = os.path.join(teacher_save_folder, "{}.pth".format(model_name))
                if self.opt.ddp:
                    to_save = model.module.state_dict()
                else:
                    to_save = model.state_dict()                     
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                    to_save['use_stereo'] = self.opt.use_stereo
                torch.save(to_save, save_path)

            # teacher_save_path = os.path.join(teacher_save_folder, "{}.pth".format("adam"))
            # torch.save(self.teacher_model_optimizer.state_dict(), teacher_save_path)

            student_save_folder = os.path.join(self.log_path, "models", "student", "weights_{}".format(self.epoch))
            if not os.path.exists(student_save_folder):
                os.makedirs(student_save_folder)

            for model_name, model in self.student_models.items():
                save_path = os.path.join(student_save_folder, "{}.pth".format(model_name))
                if self.opt.ddp:
                    to_save = model.module.state_dict()
                else:
                    to_save = model.state_dict()                     
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                    to_save['use_stereo'] = self.opt.use_stereo
                torch.save(to_save, save_path)

            student_save_path = os.path.join(student_save_folder, "{}.pth".format("adam"))
            torch.save(self.student_model_optimizer.state_dict(), student_save_path)

    def load_teacher_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        if self.local_rank == 0:
            print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if self.local_rank == 0:
                print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.teacher_models[n].state_dict()
            if self.opt.ddp:
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))            
            else:
                pretrained_dict = torch.load(path)
                
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.teacher_models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def load_student_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        if self.local_rank == 0:
            print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if self.local_rank == 0:
                print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.student_models[n].state_dict()
            if self.opt.ddp:
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))            
            else:
                pretrained_dict = torch.load(path)
                
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.student_models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.student_model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")