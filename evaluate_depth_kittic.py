from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import ECDepthOptions
import datasets
import networks
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

blur = ["brightness", "color_quant", "contrast", "dark", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
        "iso_noise", "jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]

blur_intensity = ["1", "2", "3", "4", "5"]

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256) 

def colormap(inputs, normalize=True, torch_transpose=False):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        img_ext = '.png' if opt.png else '.jpg'



        encoder = networks.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
        encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
        # encoder = networks.mpvit_base()
        # encoder.num_ch_enc = [128,224,368,480,480]
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True,allow_pickle=True, encoding='latin1')["data"]

        print("-> Evaluating")

        results_edit=open('results_kittic.txt',mode='a')
        results_edit.write("\n " + 'model_name: %s '%(opt.load_weights_folder))
        results_edit.close()

        errors_overall = 0

        for bl_category in blur:

            errors_per_bl = 0

            for bl_intensity in blur_intensity:

                data_path = os.path.join(opt.data_path, bl_category, bl_intensity, "kitti_data")

                dataset = datasets.KITTIRAWDataset(data_path, filenames,
                                                    encoder_dict['height'], encoder_dict['width'],
                                                    [0], 4, is_train=False, img_ext=img_ext)
                
                
                dataloader = DataLoader(dataset, 8, shuffle=False, num_workers=opt.num_workers,
                                        pin_memory=True, drop_last=False)

                pred_disps = []

        
                with torch.no_grad():
                    i = 0 
                    for data in dataloader:
                        input_color = data[("color", 0, 0)].cuda()

                        if opt.post_process:
                            # Post-processed results require each image to have two forward passes
                            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                        output = depth_decoder(encoder(input_color))

                        pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                        pred_disp = pred_disp.cpu()[:, 0].numpy()

                        if opt.post_process:
                            N = pred_disp.shape[0] // 2
                            pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                        # output_path = os.path.join(opt.eval_out_dir, filenames[i].split(' ')[1]+img_ext)
                        # cv2.imwrite(output_path, colormap(pred_disp)[0,:,:,::-1]*256)
                        pred_disps.append(pred_disp)
                        i = i + 8

                pred_disps = np.concatenate(pred_disps)

                if opt.eval_split == 'benchmark':
                    save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
                    print("-> Saving out benchmark predictions to {}".format(save_dir))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    for idx in range(len(pred_disps)):
                        disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
                        depth = STEREO_SCALE_FACTOR / disp_resized
                        depth = np.clip(depth, 0, 80)
                        depth = np.uint16(depth * 256)
                        save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
                        cv2.imwrite(save_path, depth)

                    print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
                    quit()

                
                if opt.eval_stereo:
                    print("   Stereo evaluation - "
                        "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
                    opt.disable_median_scaling = True
                    opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
                else:
                    print("   Mono evaluation - using median scaling")

                errors = []
                ratios = []

                for i in range(pred_disps.shape[0]):

                    gt_depth = gt_depths[i]
                    gt_height, gt_width = gt_depth.shape[:2]

                    pred_disp = pred_disps[i]
                    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                    pred_depth = 1 / pred_disp

                    if opt.eval_split == "eigen":
                        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                        0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                        crop_mask = np.zeros(mask.shape)
                        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                        mask = np.logical_and(mask, crop_mask)

                    else:
                        mask = gt_depth > 0

                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    pred_depth *= opt.pred_depth_scale_factor
                    if not opt.disable_median_scaling:
                        ratio = np.median(gt_depth) / np.median(pred_depth)
                        ratios.append(ratio)
                        pred_depth *= ratio

                    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

                    errors.append(compute_errors(gt_depth, pred_depth))

                if not opt.disable_median_scaling:
                    ratios = np.array(ratios)
                    med = np.median(ratios)
                    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

                mean_errors = np.array(errors).mean(0)

                results_edit=open('results_kittic.txt',mode='a')
                # results_edit.write("\n " + 'model_name: %s '%(opt.load_weights_folder))
                results_edit.write("\n\n " + 'blur_type: %s '%(bl_category))
                results_edit.write("\n " + 'blur_intensity: %s '%(bl_intensity))
                results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
                results_edit.close()
                print("\n  ")
                print(bl_category+'_'+bl_intensity)
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
                print("\n-> Done!")

                errors_per_bl = errors_per_bl + np.array(mean_errors) / 5

            results_edit=open('results_kittic.txt',mode='a')
            # results_edit.write("\n " + 'model_name: %s '%(opt.load_weights_folder))
            results_edit.write("\n\n " + 'errors_per_blur_type: %s '%(bl_category))
            results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*errors_per_bl.tolist()) + "\\\\")
            results_edit.close()
            errors_overall = errors_overall + np.array(errors_per_bl) / 18
        
        results_edit=open('results_kittic.txt',mode='a')
        results_edit.write("\n\n " + 'errors_overall')
        results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*errors_overall.tolist()) + "\\\\")
        results_edit.close()


if __name__ == "__main__":
    options = ECDepthOptions()
    evaluate(options.parse())
