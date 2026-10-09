import os
import time
import zipfile
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.utils.data

from .defaults import create_ddp_model
import forpt.utils.comm as comm
from forpt.datasets import build_dataset, collate_fn
from forpt.models import build_model
from forpt.utils.logger import get_root_logger
from forpt.utils.registry import Registry
from forpt.utils.misc import (
    AverageMeter,
    intersection_and_union_gpu,
    make_dirs,
)

import wandb

TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegSimpleTester(TesterBase):
    """Simple tester without fragment voting — single forward pass per sample,
    consistent with SemSegEvaluator used during training validation."""

    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            end = time.time()
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["seg_logits"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.cfg.data.num_classes,
                self.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: [{iter}/{max_iter}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    iter=i + 1,
                    max_iter=len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        if self.cfg.enable_wandb and comm.is_main_process():
            log_dict = {
                "test/mIoU": mIoU,
                "test/mAcc": mAcc,
                "test/allAcc": allAcc,
            }
            for i in range(self.cfg.data.num_classes):
                log_dict[f"test/cls_{i}-{self.cfg.data.names[i]} IoU"] = iou_class[i]
            wandb.log(
                log_dict,
                step=wandb.run.step,
            )

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsTesterSubmit(TesterBase):
    """Run classification inference on unlabeled test data and write a submission CSV."""

    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Inference (Submit) >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        self.model.eval()

        class_names = self.cfg.data.names  # list indexed by class id
        dataset = self.test_loader.dataset

        predictions = []  # list of (sample_name, pred_class_name)
        sample_idx = 0

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            preds = output_dict["cls_logits"].max(1)[1].cpu().numpy()
            batch_time.update(time.time() - end)

            for pred in preds:
                tree_id = input_dict["tree_id"].item()
                predictions.append((tree_id, class_names[pred]))
                sample_idx += 1

            logger.info(
                "Test: [{}/{}] Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                    i + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        # Gather predictions from all ranks before writing
        if comm.get_world_size() > 1:
            all_predictions = comm.all_gather(predictions)
            predictions = [item for rank_preds in all_predictions for item in rank_preds]
            # DistributedSampler pads the dataset to be evenly divisible across ranks;
            # drop the duplicated tail entries introduced by that padding.
            predictions = predictions[: len(dataset)]

        if not comm.is_main_process():
            return

        # Sort by treeID (numeric suffix of filename, e.g. test_00069 -> 69)
        predictions.sort(key=lambda x: int(x[0]))

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        csv_path = os.path.join(save_path, "predictions_test.csv")
        with open(csv_path, "w") as f:
            f.write("treeID,predicted_species\n")
            for tree_id, pred_name in predictions:
                f.write(f"{tree_id},{pred_name}\n")

        zip_path = os.path.join(save_path, "predictions_test.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, "predictions_test.csv")

        logger.info(f"Submission saved to {zip_path} ({len(predictions)} samples)")
        logger.info("<<<<<<<<<<<<<<<<< End Inference <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class RegTesterSubmit(TesterBase):
    """Run regression inference on unlabeled test data and write a Codabench submission CSV."""

    def test(self):
        import math

        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Regression Evaluation (Submit) >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        self.model.eval()

        sum_sq_err = torch.zeros(1, device="cuda")
        sum_abs_err = torch.zeros(1, device="cuda")
        sum_err = torch.zeros(1, device="cuda")
        sum_y = torch.zeros(1, device="cuda")
        sum_y2 = torch.zeros(1, device="cuda")
        count = torch.zeros(1, device="cuda")

        dataset = self.test_loader.dataset
        predictions = []  # list of (sample_id, predicted_age)

        # The hidden Codabench test set has no reference ages (NaN, or one constant
        # placeholder), so metrics and the scatter plot are only computed when the
        # split carries real reference ages (e.g. val).
        targets_np = np.array([item[1] for item in dataset.data_list], dtype=np.float64)
        has_reference = (
            targets_np.size > 0
            and np.isfinite(targets_np).all()
            and np.unique(targets_np).size > 1
        )

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)

            target = input_dict["regression"].reshape(-1).float()
            preds = output_dict["reg_logits"].reshape(-1).float()
            err = preds - target

            sum_sq_err += torch.sum(err ** 2)
            sum_abs_err += torch.sum(torch.abs(err))
            sum_err += torch.sum(err)
            sum_y += torch.sum(target)
            sum_y2 += torch.sum(target ** 2)
            count += target.numel()

            data_names = input_dict["data_name"]  # list of strings from collate_fn
            species = input_dict.get("species", [""] * len(data_names))
            for sample_id, pred, tgt, sp in zip(
                data_names, preds.cpu().numpy(), target.cpu().numpy(), species
            ):
                predictions.append((sample_id, float(pred), float(tgt), sp))

            batch_time.update(time.time() - end)
            progress = "Test: [{}/{}] Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                i + 1, len(self.test_loader), batch_time=batch_time
            )
            if has_reference:
                progress += " MAE {:.4f}".format((sum_abs_err / count).item())
            logger.info(progress)

        if comm.get_world_size() > 1:
            dist.all_reduce(sum_sq_err)
            dist.all_reduce(sum_abs_err)
            dist.all_reduce(sum_err)
            dist.all_reduce(sum_y)
            dist.all_reduce(sum_y2)
            dist.all_reduce(count)
            all_predictions = comm.all_gather(predictions)
            # DistributedSampler pads the dataset to be evenly divisible; dedupe by
            # sample_id since padding duplicates are identical predictions.
            merged = {}
            for rank_preds in all_predictions:
                for sample_id, pred, tgt, sp in rank_preds:
                    merged.setdefault(sample_id, (pred, tgt, sp))
            predictions = [(sid, v[0], v[1], v[2]) for sid, v in merged.items()]

        n = count.item()
        mse = (sum_sq_err / count).item() if n > 0 else float("nan")
        rmse = math.sqrt(mse) if not math.isnan(mse) else float("nan")
        mae = (sum_abs_err / count).item() if n > 0 else float("nan")
        mbe = (sum_err / count).item() if n > 0 else float("nan")
        ss_res = sum_sq_err.item()
        ss_tot = (sum_y2 - (sum_y ** 2) / count).item() if n > 0 else float("nan")
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        if comm.is_main_process() and not has_reference:
            logger.info(
                "Reference ages look like placeholders; skipping metrics. "
                "Submit predictions_test.zip to Codabench for scoring."
            )
        elif comm.is_main_process():
            logger.info(
                "Test result: MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MBE: {:.4f}, R2: {:.4f}".format(
                    mse, rmse, mae, mbe, r2
                )
            )
            if self.cfg.enable_wandb:
                wandb.log(
                    {
                        "test/MSE": mse,
                        "test/RMSE": rmse,
                        "test/MAE": mae,
                        "test/MBE": mbe,
                        "test/R2": r2,
                    }
                )

        if not comm.is_main_process():
            return

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        csv_path = os.path.join(save_path, "predictions_test.csv")
        with open(csv_path, "w") as f:
            f.write("sampleID,age_pred\n")
            for sample_id, age_pred, _, _ in predictions:
                f.write(f"{sample_id},{age_pred}\n")

        zip_path = os.path.join(save_path, "predictions_test.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, "predictions_test.csv")

        logger.info(f"Submission saved to {zip_path} ({len(predictions)} samples)")

        # Scatter plot of predicted vs reference age, coloured by species.
        if has_reference:
            plot_path = self.plot_pred_vs_ref(predictions, rmse, mbe, r2, save_path)
            logger.info(f"Scatter plot saved to: {plot_path}")
            if self.cfg.enable_wandb:
                wandb.log({"test/pred_vs_ref": wandb.Image(plot_path)})

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def plot_pred_vs_ref(predictions, rmse, mbe, r2, save_path):
        """Scatter of predicted vs reference age with a dashed 1:1 line, coloured by species.

        predictions: list of (sample_id, pred, target, species).
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_ids = np.array([sid for sid, _, _, _ in predictions])
        preds = np.array([p for _, p, _, _ in predictions], dtype=np.float64)
        targets = np.array([t for _, _, t, _ in predictions], dtype=np.float64)
        species_arr = np.array([str(s) for _, _, _, s in predictions])

        # save raw values so plots for all benchmarked methods can be regenerated later
        np.savez(
            os.path.join(save_path, "pred_vs_ref.npz"),
            sample_ids=sample_ids,
            preds=preds,
            targets=targets,
            species=species_arr,
        )

        # display name and colour per raw species label in the CSV
        species_style = {
            "pine": ("Pinus sylvestris", "#e08a3c"),
            "spruce": ("Picea abies", "#5aa2dd"),
        }

        fig, ax = plt.subplots(figsize=(6, 6))
        plotted = set()
        for key, (label, color) in species_style.items():
            mask = species_arr == key
            if not mask.any():
                continue
            ax.scatter(
                targets[mask], preds[mask], s=28, c=color, alpha=0.75,
                edgecolors="none", label=label,
            )
            plotted.add(key)
        # anything without a known species label -> neutral grey, unlabeled
        other = ~np.isin(species_arr, list(plotted))
        if other.any():
            ax.scatter(
                targets[other], preds[other], s=28, c="#999999", alpha=0.6,
                edgecolors="none",
            )

        lim_max = max(targets.max(), preds.max()) * 1.05 if targets.size > 0 else 1.0
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect("equal")
        ax.set_xlabel("Reference Age (years)", fontsize=13)
        ax.set_ylabel("Predicted Age (years)", fontsize=13)
        ax.text(
            0.03, 0.97,
            "RMSE = {:.0f} years\nMD = {:.0f} years\n$R^2$ = {:.2f}".format(rmse, mbe, r2),
            transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
        )
        if plotted:
            leg = ax.legend(
                loc="upper left", bbox_to_anchor=(0.03, 0.80), frameon=False,
                fontsize=12, handletextpad=0.3,
            )
            for txt in leg.get_texts():
                txt.set_fontstyle("italic")
        fig.tight_layout()
        plot_path = os.path.join(save_path, "pred_vs_ref_scatter.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        return plot_path

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
