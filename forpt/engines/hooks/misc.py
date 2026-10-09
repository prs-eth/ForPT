import sys
import os
import shutil
import time
import wandb
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from forpt.utils.timer import Timer
from forpt.utils.comm import is_main_process
import forpt.utils.comm as comm
from forpt.engines.test import TESTERS

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="global_step")
            wandb.define_metric("train_batch/*", step_metric="global_step")
            wandb.define_metric("train/*", step_metric="epoch")

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info


    def compute_total_grad_norm(self, parameters, norm_type=2):
        parameters = [p for p in parameters if p.grad is not None]
        norm_list = [p.grad.detach().norm(norm_type) for p in parameters]
        total_norm = torch.norm(torch.stack(norm_list), norm_type)
        return total_norm

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:

                if hasattr(self.trainer.model.module, "mask_size"):
                    wandb.log(
                        {"global_step": self.curr_iter, 
                        "params/lr": lr,
                        "params/norm": self.compute_total_grad_norm(self.trainer.model.parameters(), norm_type=2.0),
                        "params/mask_size": self.trainer.model.module.mask_size,
                        "params/mask_ratio": self.trainer.model.module.mask_ratio,
                        "params/teacher_temp": self.trainer.model.module.teacher_temp,
                        "params/ema_momentum": self.trainer.model.module.momentum,
                        "params/weight_decay": self.trainer.optimizer.state_dict()["param_groups"][0]["weight_decay"]
                        }, step=self.curr_iter
                    )
                else:
                    wandb.log(
                        {"global_step": self.curr_iter, 
                        "params/lr": lr,
                        "params/norm": self.compute_total_grad_norm(self.trainer.model.parameters(), norm_type=2.0),
                        }, step=self.curr_iter
                    )

                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "global_step": self.curr_iter,
                            f"train_batch/{key}": self.trainer.storage.history(key).val,
                        },
                        step=wandb.run.step,
                    )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            if self.trainer.cfg.enable_wandb:

                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "epoch": self.trainer.epoch + 1,
                            f"train/{key}": self.trainer.storage.history(key).avg,
                        },
                        step=wandb.run.step,
                    )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate and 'current_metric_value' in self.trainer.comm_info:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                higher_is_better = current_metric_name not in ["MSE", "RMSE", "MAE", "MBE"]
                # if current_metric_value > self.trainer.best_metric_value:
                if (
                    (higher_is_better and current_metric_value > self.trainer.best_metric_value)
                    or
                    ((not higher_is_better) and current_metric_value < self.trainer.best_metric_value)
                    or 
                    self.trainer.best_metric_value in [float("-inf"), float("inf")]
                ):
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(f"Higher is better: {higher_is_better}")
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")



@HOOKS.register_module()
class ForestFormerCheckpointLoader(HookBase):
    """Load a backbone pretrained with ForestFormer3D (fully supervised
    instance + semantic segmentation, mmdet3d/mmengine-style checkpoint).
    """

    def __init__(self, keywords="unet.", replacement="backbone.", strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading ForestFormer3D pretrained weight ...")
        if not (self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight)):
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
            return

        self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
        checkpoint = torch.load(
            self.trainer.cfg.weight,
            map_location=lambda storage, loc: storage.cuda(),
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.trainer.logger.info(
            f"Keeping layer weights with keyword: {self.keywords}, "
            f"replace keyword with: {self.replacement}"
        )

        weight = OrderedDict()
        skipped = 0
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if not key.startswith(self.keywords):
                skipped += 1
                continue
            key = self.replacement + key[len(self.keywords) :]
            if comm.get_world_size() > 1:
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value

        self.trainer.logger.info(
            f"Remapped {len(weight)} backbone weights, "
            f"skipped {skipped} non-backbone weights."
        )
        missing, unexpected = self.trainer.model.load_state_dict(
            weight, strict=self.strict
        )
        self.trainer.logger.info(
            f"Loaded {len(weight) - len(unexpected)} weights into the model."
        )
        self.trainer.logger.info(f"Missing keys ({len(missing)}): {missing}")
        # With enc_mode=True the model has no decoder, so all backbone.dec.* keys
        # land here; list them compactly to keep the log readable.
        self.trainer.logger.info(
            f"Unexpected keys ({len(unexpected)}): "
            f"{unexpected[:5] + ['...'] if len(unexpected) > 5 else unexpected}"
        )

        if self.trainer.cfg.resume:
            # A ForestFormer3D checkpoint carries an mmengine optimizer / scheduler
            # state that does not match this trainer, so it cannot be resumed from.
            self.trainer.logger.warning(
                "resume=True is not supported by ForestFormerCheckpointLoader; "
                "the weight is used for initialization only. Use CheckpointLoader "
                "with a checkpoint saved by this trainer to resume."
            )


@HOOKS.register_module()
class CheckpointLoaderList(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()

            
    
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                for i in range(len(self.keywords)):
                    if self.keywords[i] in key:
                        key = key.replace(self.keywords[i], self.replacement[i], 1)
                        if comm.get_world_size() == 1:
                            key = key[7:]  # module.xxx.xxx -> xxx.xxx
                        weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path, weights_only=False)
            state_dict = checkpoint["state_dict"]
            tester.model.load_state_dict(state_dict, strict=True)
        tester.test()
