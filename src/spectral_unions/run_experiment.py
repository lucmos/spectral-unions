import tempfile

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from nn_core.common.utils import get_env

# TODO: DELETE THIS FILE

register = ...
load_config = ...
CustomMLFlowLogger = ...
CustomWandbLogger = ...

seed_everything(0)
config_name = "experiment.yml"
torch.multiprocessing.set_sharing_strategy("file_system")

cfg = load_config(config_name)

model_class = register.get_model_class(cfg["params"]["model"]["class"])
model_params = (
    cfg["params"]["model"]["params"]
    if "params" in cfg["params"]["model"] and cfg["params"]["model"]["params"] is not None
    else {}
)
if (
    "weight_inizialization" in cfg["params"]["training"]
    and cfg["params"]["training"]["weight_inizialization"] is not None
):
    model = model_class.load_from_checkpoint(
        cfg["params"]["training"]["weight_inizialization"],
        hparams=cfg,
        **cfg["params"]["model"]["params"],
    )
else:
    model = model_class(cfg, **model_params)


callbacks = []
if cfg["params"]["logger"]["active"]:
    mlf_logger = CustomMLFlowLogger(
        experiment_name=cfg["experiment_name"],
        tracking_uri=f'file:{get_env("MLFLOW_DATASTORE")}',
    )

    wandb_logger = CustomWandbLogger(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        log_model=True,
        tags=cfg["tags"],
        save_dir="/run/media/luca/Storage/wandb",
    )
    # wandb_logger.experiment.watch(
    #     model, log="all", log_freq=cfg["params"]["logger"]["log_every_n_steps"]
    # )
    wandb_logger.experiment.watch(model, log_freq=cfg["params"]["logger"]["log_every_n_steps"])
    wandb_logger.log_hyperparams({"mlflow_run_id": mlf_logger.run_id})
    callbacks.append(LearningRateMonitor())


else:
    mlf_logger = None
    wandb_logger = None
# early_stopping = EarlyStopping(
#     monitor=cfg["params"]["early_stop"]["monitor_metric"],
#     min_delta=cfg["params"]["early_stop"]["min_delta"],
#     patience=cfg["params"]["early_stop"]["patience"],
#     mode=cfg["params"]["early_stop"]["mode"],
#     verbose=True,
#     strict=True,
# )


trainer = pl.Trainer(
    callbacks=callbacks,
    fast_dev_run=False,  # todo: debug flag!
    gpus=cfg["params"]["training"]["gpus"],
    precision=cfg["params"]["training"]["precision"],
    log_gpu_memory=False,  # cfg["params"]["logger"]["log_gpu_memory"],
    progress_bar_refresh_rate=cfg["params"]["logger"]["progress_bar_refresh_rate"],
    max_epochs=cfg["params"]["training"]["max_epochs"],
    min_epochs=cfg["params"]["training"]["min_epochs"],
    overfit_batches=cfg["params"]["training"]["overfit_batches"],
    num_sanity_val_steps=5,
    check_val_every_n_epoch=cfg["params"]["training"]["check_val_every_n_epoch"],
    val_check_interval=cfg["params"]["training"]["val_check_interval"],
    logger=[mlf_logger, wandb_logger] if mlf_logger is not None and wandb_logger is not None else None,
    log_every_n_steps=cfg["params"]["logger"]["log_every_n_steps"],
    flush_logs_every_n_steps=cfg["params"]["logger"]["flush_logs_every_n_steps"],
    resume_from_checkpoint=cfg["params"]["training"]["resume_from_checkpoint"],
    profiler=False,  # AdvancedProfiler(),
    limit_train_batches=cfg["params"]["training"]["limit_train_batches"],
    limit_val_batches=cfg["params"]["training"]["limit_val_batches"],
)

trainer.fit(
    model,
)

if mlf_logger is not None:
    model_name = f"{tempfile.mkdtemp()}/weights_epoch{trainer.current_epoch:04d}.tar"
    trainer.save_checkpoint(model_name)
    mlf_logger.experiment.log_artifact(run_id=mlf_logger.run_id, local_path=model_name, artifact_path="weights")
