import os
import math
import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import forpt.utils.comm as comm
from forpt.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS

from forpt.datasets.transform import Compose
from forpt.utils.misc import knn_segmentor, measure_similarity_partial, measure_similarity_full, measure_similarity
from forpt.metrics.semantic import ConfusionMatrix

from torch_scatter import scatter_mean
from torch.nn.functional import normalize
from tqdm import tqdm

import torch.nn.functional as F

@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def __init__(self, eval_interval=1):
        self.eval_interval = eval_interval

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.trainer.epoch % self.eval_interval == 0:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)

            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )

                for i in range(self.trainer.cfg.data.num_classes):
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Acc": acc_class[
                                i
                            ],
                        },
                        step=wandb.run.step,
                    )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=True):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        # self.eval()
        # import sys
        # sys.exit()
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():

                # output_dict = self.trainer.model(input_dict)
                if hasattr(self.trainer, 'pca'):
                    output_dict = self.trainer.model(input_dict, self.trainer.pca)
                else:
                    output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            # if "origin_coord" in input_dict.keys():
            #     idx, _ = pointops.knn_query(
            #         1,
            #         input_dict["coord"].float(),
            #         input_dict["offset"].int(),
            #         input_dict["origin_coord"].float(),
            #         input_dict["origin_offset"].int(),
            #     )
            #     pred = pred[idx.flatten().long()]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class KNNSegEvaluator_forest(HookBase):
    def __init__(self, data_root="data/forpt_ssl_corpus/knn_track", write_cls_iou=False):
        self.data_root = data_root
        self.write_cls_iou = write_cls_iou
        # knn hyper-parameters
        self.T = 0.07
        self.nb_knn = [1, 10, 20, 50, 100, 200]

        self.train_scene_list = [
            'MLS_Plot_82_panoptic_train', 'NIBIO2_plot25_annotated_train', 
            'NIBIO2_plot16_annotated_train', 'NIBIO_plot_10_annotated_train', 
            'NIBIO2_plot12_annotated_train', 'NIBIO_plot_6_annotated_train', 
            'NIBIO_plot_21_annotated_train', 'NIBIO2_plot42_annotated_train', 
            'NIBIO2_plot46_annotated_train', 'NIBIO2_plot23_annotated_train', 
            'MLS_Plot_57_panoptic_train', 
            'NIBIO_plot_7_annotated_train', 'RMIT_train_train', 'NIBIO2_plot2_annotated_train', 
            'NIBIO2_plot9_annotated_train', 'NIBIO_plot_11_annotated_train', 
            'NIBIO2_plot38_annotated_train', 'NIBIO_plot_4_annotated_train', 
            'NIBIO2_plot40_annotated_train', 'NIBIO2_plot39_annotated_train', 
            'NIBIO2_plot33_annotated_train', 'SCION_plot_39_annotated_train', 
            'NIBIO_plot_3_annotated_train', 'NIBIO2_plot57_annotated_train', 
            'TUWIEN_train_train', 'NIBIO2_plot61_annotated_train', 'NIBIO2_plot28_annotated_train', 
            'NIBIO2_plot55_annotated_train', 'NIBIO2_plot56_annotated_train', 
            'NIBIO2_plot21_annotated_train', 'NIBIO2_plot22_annotated_train', 
            'SCION_plot_87_annotated_train', 'NIBIO2_plot24_annotated_train', 
            'NIBIO_plot_12_annotated_train', 'NIBIO2_plot43_annotated_train', 
            'CULS_plot_1_annotated_train', 'NIBIO2_plot31_annotated_train', 
            'NIBIO2_plot20_annotated_train', 'NIBIO2_plot8_annotated_train', 
            'NIBIO2_plot45_annotated_train', 'NIBIO2_plot14_annotated_train', 
            'NIBIO2_plot51_annotated_train', 'NIBIO2_plot26_annotated_train', 
            'MLS_Plot_56_panoptic_train', 'MLS_Plot_107_panoptic_train', 
            'NIBIO2_plot50_annotated_train',
            # 'dls_merged_230209_panoptic_train', 
            ]


        self.val_scene_list = [
            'NIBIO_plot_8_annotated_val', 'MLS_burumPlot_2_panoptic_val', 
            'NIBIO2_plot47_annotated_val', 'CULS_plot_3_annotated_val', 
            'NIBIO_plot_16_annotated_val', 'NIBIO2_plot54_annotated_val', 
             'NIBIO2_plot41_annotated_val', 
            'NIBIO_plot_9_annotated_val', 'NIBIO_plot_19_annotated_val', 
            'SCION_plot_35_annotated_val', 'NIBIO2_plot59_annotated_val', 
            'NIBIO2_plot44_annotated_val', 'NIBIO2_plot13_annotated_val', 
            'NIBIO_plot_13_annotated_val', 'NIBIO_plot_2_annotated_val',
            # 'dls_merged_230209_panoptic_val'
            ]


        self.val_features_last = None
        self.log_begin = True
        

    def before_train(self):
        self.knn_transform = Compose(self.trainer.cfg.knn_transform)
        self.val_cm = ConfusionMatrix(self.trainer.cfg.data.num_classes)
        self.val_cm.cuda()



        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="epoch")
        
        dist.barrier()
        if dist.get_rank() == 0:
            self.eval()
        dist.barrier()

    def after_epoch(self):
        if (self.trainer.epoch+1) % self.trainer.cfg.knn_evaluate_interval == 0:
            dist.barrier()
            # print('current rank: ', dist.get_rank())
            if dist.get_rank() == 0:
                self.eval()
            dist.barrier()

    def eval(self):


        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start KNN Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        ### extract features for all training data
        train_feats_bank = []
        train_labels_bank = []

        for i, scene_name in enumerate(tqdm(self.train_scene_list)):
            scene_data = np.load(os.path.join(self.data_root, scene_name+'.npy'),allow_pickle=True).tolist()

            segment = scene_data['label'].argmax(-1)
            superlabel = scene_data['superlabel']
            superindex = scene_data['superindex']
            segment[segment==self.trainer.cfg.data.num_classes] = self.trainer.cfg.data.ignore_index
            point = {'coord': scene_data['coord'],
                'segment': segment,
                }

            input_dict = self.knn_transform(point)

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                point = self.trainer.model.module.teacher.backbone(input_dict)

                point = self.trainer.model.module.up_cast(point, self.trainer.model.module.up_cast_level_infer)
                while "pooling_parent" in point.keys():
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = point.feat[inverse]
                    point = parent

                voxel_feat = point.feat[point.inverse]
                
                # aggragate superpoint features
                feat = scatter_mean(voxel_feat, torch.from_numpy(superindex).to(voxel_feat.device),dim=0)

                feat = normalize(feat, dim=1, p=2)
        
                train_feats_bank.append(feat.detach().to(torch.device("cpu")))
                train_labels_bank.append(torch.from_numpy(superlabel))


        train_features = torch.cat(train_feats_bank).to(voxel_feat.device)
        
        train_labels_hist = torch.cat(train_labels_bank).to(voxel_feat.device)

        train_labels_scalar = train_labels_hist.argmax(-1)
        valid_mask = train_labels_scalar != self.trainer.cfg.data.num_classes
        train_features = train_features[valid_mask]
        train_labels_hist = train_labels_hist[valid_mask]
        train_labels_scalar = train_labels_scalar[valid_mask]

        ### knn segmentation for validation scenes
        val_feats_bank = []
        for i, scene_name in enumerate(tqdm(self.val_scene_list)):
            scene_data = np.load(os.path.join(self.data_root, scene_name+'.npy'),allow_pickle=True).tolist()

            segment = scene_data['label'].argmax(-1)
            superlabel = scene_data['superlabel']
            superindex = scene_data['superindex']
            segment[segment==self.trainer.cfg.data.num_classes] = self.trainer.cfg.data.ignore_index
            point = {'coord': scene_data['coord'],
                'segment': segment,
                }

            input_dict = self.knn_transform(point)

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                point = self.trainer.model.module.teacher.backbone(input_dict)

                point = self.trainer.model.module.up_cast(point, self.trainer.model.module.up_cast_level_infer)
                while "pooling_parent" in point.keys():
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = point.feat[inverse]
                    point = parent

            voxel_feat = point.feat[point.inverse]
            # aggragate superpoint features
            feat = scatter_mean(voxel_feat, torch.from_numpy(superindex).to(voxel_feat.device),dim=0)
            feat = normalize(feat, dim=1, p=2)
            val_feats_bank.append(feat.detach().to(torch.device("cpu")))
            probas_for_k = knn_segmentor(feat, train_features, train_labels_hist, self.trainer.cfg.data.num_classes, self.T, self.nb_knn)

            # use prediction and gt at superpoint level to compute miou
            pred = probas_for_k[20].argmax(-1)
            segment = torch.from_numpy(superlabel).to(pred.device)


            self.val_cm(pred, segment)
            del probas_for_k


        epoch_cm = self.val_cm

        m_iou = epoch_cm.miou().tolist()
        all_acc = epoch_cm.oa().tolist() # actually oa
        m_acc = epoch_cm.macc().tolist()
        iou_class = epoch_cm.iou()[0].tolist()



        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou {iou:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    # accuracy=acc_class[i],
                )
            )
        if self.trainer.epoch == 0 and self.log_begin:
            current_epoch = 0
            self.log_begin = False
        else:
            current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            # self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "epoch": current_epoch,
                        "val/miou": m_iou,
                        "val/macc": m_acc,
                        "val/allacc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "epoch": current_epoch,
                                f"val/iou_{self.trainer.cfg.data.names[i]}": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

        self.val_cm.reset()
        epoch_cm.reset()

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class RegEvaluator(HookBase):

    def __init__(self, train_eval_freq=1):
        # run the diagnostic train-set pass every `train_eval_freq` epochs
        self.train_eval_freq = max(1, int(train_eval_freq))
        self._train_eval_loader = None
        self._train_eval_built = False

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
            wandb.define_metric("train_eval/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()
            # Optional diagnostic pass over the (un-augmented) training set. Enabled
            # by adding a `data.train_eval` dataset to the config (train split with
            # val-style transforms). Logged under train_eval/* — does NOT affect the
            # best-checkpoint metric, which comes from val only. Runs every
            # `train_eval_freq` epochs to limit overhead.
            if (self.trainer.epoch + 1) % self.train_eval_freq == 0:
                try:
                    train_eval_loader = self._get_train_eval_loader()
                    if train_eval_loader is not None:
                        self.eval(loader=train_eval_loader, prefix="train_eval",
                                  update_best=False)
                except Exception as e:  # diagnostic must never kill training
                    self.trainer.logger.warning(f"train_eval pass failed: {e}")

    def _get_train_eval_loader(self):
        if self._train_eval_built:
            return self._train_eval_loader
        self._train_eval_built = True
        data_cfg = self.trainer.cfg.data.get("train_eval", None)
        if data_cfg is None:
            self._train_eval_loader = None
            return None
        from forpt.datasets import build_dataset, collate_fn
        ds = build_dataset(data_cfg)
        sampler = (
            torch.utils.data.distributed.DistributedSampler(ds)
            if comm.get_world_size() > 1 else None
        )
        # num_workers=0: this loader is built lazily inside the training loop, after
        # CUDA is initialized in the parent. Forking DataLoader workers at that point
        # triggers "CUDA initialization error". The pass is tiny (cached CPU data),
        # so single-process loading is fine.
        self._train_eval_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.trainer.cfg.batch_size_val_per_gpu,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        self.trainer.logger.info(
            f"[RegEvaluator] built train_eval loader ({len(ds)} samples)"
        )
        return self._train_eval_loader

    def eval(self, loader=None, prefix="val", update_best=True):
        if loader is None:
            loader = self.trainer.val_loader
        self.trainer.logger.info(
            f">>>>>>>>>>>>>>>> Start Regression Evaluation [{prefix}] >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        # Global accumulators for regression metrics
        sum_sq_err = torch.zeros(1, device="cuda")
        sum_abs_err = torch.zeros(1, device="cuda")
        sum_err = torch.zeros(1, device="cuda")
        sum_y = torch.zeros(1, device="cuda")
        sum_y2 = torch.zeros(1, device="cuda")
        count = torch.zeros(1, device="cuda")
        loss_sum = torch.zeros(1, device="cuda")
        loss_count = torch.zeros(1, device="cuda")

        # per-sample buffers for the scatter plot (needs the dataset to carry
        # "species"; set inference_mode=True on that dataset so it does)
        preds_buf, targets_buf, species_buf = [], [], []

        for i, input_dict in enumerate(loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            target = input_dict["regression"].reshape(-1).float()
            preds = output_dict["reg_logits"].reshape(-1).float()
            loss = output_dict["loss"].detach()

            preds_buf.append(preds.detach().cpu().numpy())
            targets_buf.append(target.detach().cpu().numpy())
            sp = input_dict.get("species", None)
            species_buf.extend(list(sp) if sp is not None else [""] * target.numel())

            err = preds - target

            sum_sq_err += torch.sum(err ** 2)
            sum_abs_err += torch.sum(torch.abs(err))
            sum_err += torch.sum(err)
            sum_y += torch.sum(target)
            sum_y2 += torch.sum(target ** 2)
            count += target.numel()

            loss_sum += loss
            loss_count += 1

            if comm.get_world_size() > 1:
                # optional per-iteration logging can stay local, but is often noisy
                pass

            if comm.is_main_process():
                self.trainer.logger.info(
                    "Test: [{iter}/{max_iter}] Loss {loss:.4f}".format(
                        iter=i + 1, max_iter=len(loader), loss=loss.item()
                    )
                )

        if comm.get_world_size() > 1:
            dist.all_reduce(sum_sq_err)
            dist.all_reduce(sum_abs_err)
            dist.all_reduce(sum_err)
            dist.all_reduce(sum_y)
            dist.all_reduce(sum_y2)
            dist.all_reduce(count)
            dist.all_reduce(loss_sum)
            dist.all_reduce(loss_count)

        count_value = count.item()
        loss_avg = (loss_sum / loss_count).item() if loss_count.item() > 0 else float("nan")

        mse = (sum_sq_err / count).item() if count_value > 0 else float("nan")
        rmse = math.sqrt(mse) if not math.isnan(mse) else float("nan")
        mae = (sum_abs_err / count).item() if count_value > 0 else float("nan")
        mbe = (sum_err / count).item() if count_value > 0 else float("nan")

        # SS_tot = sum((y - mean(y))^2) = sum(y^2) - (sum(y)^2 / n)
        ss_res = sum_sq_err.item()
        ss_tot = (sum_y2 - (sum_y ** 2) / count).item() if count_value > 0 else float("nan")
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        if comm.is_main_process():
            self.trainer.logger.info(
                "{} result: Loss Avg: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, "
                "MAE: {:.4f}, MBE: {:.4f}, R2: {:.4f}".format(
                    prefix.capitalize(), loss_avg, mse, rmse, mae, mbe, r2
                )
            )

            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(f"{prefix}/loss", loss_avg, current_epoch)
                self.trainer.writer.add_scalar(f"{prefix}/MSE", mse, current_epoch)
                self.trainer.writer.add_scalar(f"{prefix}/RMSE", rmse, current_epoch)
                self.trainer.writer.add_scalar(f"{prefix}/MAE", mae, current_epoch)
                self.trainer.writer.add_scalar(f"{prefix}/MBE", mbe, current_epoch)
                self.trainer.writer.add_scalar(f"{prefix}/R2", r2, current_epoch)

                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            f"{prefix}/loss": loss_avg,
                            f"{prefix}/MSE": mse,
                            f"{prefix}/RMSE": rmse,
                            f"{prefix}/MAE": mae,
                            f"{prefix}/MBE": mbe,
                            f"{prefix}/R2": r2,
                        }
                    )

        # gather per-sample predictions across ranks and save the scatter plot.
        # Wrapped so a plotting/gather error can never kill training. Note
        # comm.gather returns an empty list on non-main ranks, so use `or [...]`.
        try:
            preds_np = np.concatenate(preds_buf) if len(preds_buf) else np.zeros(0)
            targets_np = np.concatenate(targets_buf) if len(targets_buf) else np.zeros(0)
            species_np = np.array(species_buf, dtype=object)
            if comm.get_world_size() > 1:
                preds_np = np.concatenate(comm.gather(preds_np, dst=0) or [np.zeros(0)])
                targets_np = np.concatenate(comm.gather(targets_np, dst=0) or [np.zeros(0)])
                species_np = np.concatenate(
                    comm.gather(species_np, dst=0) or [np.zeros(0, dtype=object)]
                )
            if comm.is_main_process() and targets_np.size > 0:
                self._plot_val_scatter(
                    preds_np, targets_np, species_np,
                    rmse, mbe, r2, self.trainer.epoch + 1, prefix,
                )
        except Exception as e:  # never let plotting kill training
            self.trainer.logger.warning(f"{prefix} scatter plot failed: {e}")

        # only the val pass drives the best-checkpoint metric
        if update_best:
            self.trainer.comm_info["current_metric_value"] = rmse
            self.trainer.comm_info["current_metric_name"] = "RMSE"
        self.trainer.logger.info(
            f"<<<<<<<<<<<<<<<<< End Regression Evaluation [{prefix}] <<<<<<<<<<<<<<<<<")

    def _plot_val_scatter(self, preds, targets, species, rmse, mbe, r2, epoch, prefix="val"):
        """Save a predicted-vs-reference scatter for the given split at this epoch."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        preds = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        species = np.array([str(s) for s in species]) if len(species) == len(targets) \
            else np.array([""] * len(targets))

        species_style = {
            "pine": ("Pinus sylvestris", "#e08a3c"),
            "spruce": ("Picea abies", "#5aa2dd"),
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        plotted = set()
        for key, (label, color) in species_style.items():
            m = species == key
            if not m.any():
                continue
            ax.scatter(targets[m], preds[m], s=28, c=color, alpha=0.75,
                       edgecolors="none", label=label)
            plotted.add(key)
        other = ~np.isin(species, list(plotted))
        if other.any():
            ax.scatter(targets[other], preds[other], s=28, c="#999999",
                       alpha=0.6, edgecolors="none")

        lim_max = max(targets.max(), preds.max()) * 1.05 if targets.size else 1.0
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect("equal")
        ax.set_xlabel("Reference Age (years)", fontsize=13)
        ax.set_ylabel("Predicted Age (years)", fontsize=13)
        title_split = {"val": "Validation", "train_eval": "Train"}.get(prefix, prefix)
        ax.set_title(f"{title_split} — epoch {epoch}", fontsize=12)
        ax.text(0.03, 0.97,
                "RMSE = {:.0f} years\nMD = {:.0f} years\n$R^2$ = {:.2f}".format(rmse, mbe, r2),
                transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
        if plotted:
            leg = ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.80),
                            frameon=False, fontsize=12, handletextpad=0.3)
            for txt in leg.get_texts():
                txt.set_fontstyle("italic")
        fig.tight_layout()

        out_dir = os.path.join(self.trainer.cfg.save_path, f"{prefix}_scatter")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def after_train(self):
        if "current_metric_name" in self.trainer.comm_info:
            self.trainer.logger.info(
                "Best {}: {:.4f}".format(
                    self.trainer.comm_info["current_metric_name"],
                    self.trainer.best_metric_value
                )
            )
        else:
            self.trainer.logger.info(
                "Best {}: {:.4f}".format("MSE", self.trainer.best_metric_value)
            )
