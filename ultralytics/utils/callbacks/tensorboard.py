# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

try:
    # WARNING: do not move SummaryWriter import due to protobuf bug https://github.com/ultralytics/ultralytics/pull/4674
    from torch.utils.tensorboard import SummaryWriter

    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["tensorboard"] is True  # verify integration is enabled
    WRITER = None  # TensorBoard SummaryWriter instance
    PREFIX = colorstr("TensorBoard: ")

    # Imports below only required if TensorBoard enabled
    import warnings
    from copy import deepcopy

    from ultralytics.utils.torch_utils import de_parallel, torch

except (ImportError, AssertionError, TypeError, AttributeError):
    # TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows
    # AttributeError: module 'tensorflow' has no attribute 'io' if 'tensorflow' not installed
    SummaryWriter = None


def _log_scalars(scalars, step=0):
    """Logs scalar values to TensorBoard."""
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


def _log_images(path_prefix, images, step=0):
    """Log image batches to TensorBoard.

    Supports BCHW tensors in [0,1] and logs up to 4 images.
    """
    if WRITER is None or images is None:
        return
    if not torch.is_tensor(images):
        return
    if images.ndim != 4:
        return
    count = min(4, int(images.shape[0]))
    imgs = images[:count].detach().cpu()
    imgs = imgs.clamp(0, 1)
    for i in range(count):
        WRITER.add_image(f"{path_prefix}/img_{i}", imgs[i], step)


def _log_model_histograms(trainer, step=0):
    """Log lightweight parameter histograms for monitoring drift/stability."""
    if WRITER is None:
        return
    model = de_parallel(trainer.model)
    logged = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.data.numel() < 32:
            continue
        WRITER.add_histogram(f"params/{name}", p.data.detach().cpu(), step)
        logged += 1
        if logged >= 24:
            break


def _log_train_val_pair(trainer, step=0):
    """Log paired train-vs-val scalars for quick comparison panels in TensorBoard."""
    if WRITER is None:
        return
    tl = trainer.label_loss_items(trainer.tloss, prefix="train")
    for key in ("box_loss", "cls_loss", "dfl_loss", "branch_div_loss"):
        tk = f"train/{key}"
        vk = f"val/{key}"
        if tk in tl and vk in trainer.metrics:
            WRITER.add_scalars(f"compare/{key}", {"train": tl[tk], "val": trainer.metrics[vk]}, step)
            WRITER.add_scalar(f"compare/{key}/train", tl[tk], step)
            WRITER.add_scalar(f"compare/{key}/val", trainer.metrics[vk], step)


def _log_tensorboard_graph(trainer):
    """Log model graph to TensorBoard."""
    # Input image
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())  # for device, type
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # suppress jit trace warning
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)  # suppress jit trace warning

        model_ref = de_parallel(trainer.model)
        has_y13_dynamic = any(
            m.__class__.__name__ in {"AAttn", "AdaHGComputation", "AdaHGConv", "C3AH", "FullPAD_Tunnel", "HyperACE"}
            for m in model_ref.modules()
        )
        if has_y13_dynamic:
            model_ref.eval()
            with torch.no_grad():
                _ = model_ref(im)
            LOGGER.info(f"{PREFIX}model graph trace skipped (dynamic YOLOv13 modules detected), scalars only ✅")
            return

        # Try simple method first (YOLO)
        try:
            trainer.model.eval()  # place in .eval() mode to avoid BatchNorm statistics changes
            WRITER.add_graph(torch.jit.trace(model_ref, im, strict=False), [])
            LOGGER.info(f"{PREFIX}model graph visualization added ✅")
            return

        except Exception:
            # Fallback to TorchScript export steps (RTDETR)
            try:
                model = deepcopy(de_parallel(trainer.model))
                model.eval()
                model = model.fuse(verbose=False)
                for m in model.modules():
                    if hasattr(m, "export"):  # Detect, RTDETRDecoder (Segment and Pose use Detect base class)
                        m.export = True
                        m.format = "torchscript"
                model(im)  # dry run
                WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
                LOGGER.info(f"{PREFIX}model graph visualization added ✅")
            except Exception as e:
                LOGGER.warning(f"{PREFIX}WARNING ⚠️ TensorBoard graph visualization failure {e}")


def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}")


def on_train_start(trainer):
    """Log TensorBoard graph."""
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer):
    """Logs scalar statistics at the end of a training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 0:
        batch = getattr(trainer, "batch", None)
        imgs = batch.get("img") if isinstance(batch, dict) else None
        _log_images("train/samples", imgs, trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    _log_scalars(trainer.metrics, trainer.epoch + 1)
    if WRITER:
        WRITER.add_text("tb_debug/on_fit_epoch_end", f"epoch={trainer.epoch + 1}", trainer.epoch + 1)
    _log_train_val_pair(trainer, trainer.epoch + 1)
    if trainer.epoch % 10 == 0:
        _log_model_histograms(trainer, trainer.epoch + 1)


def on_train_end(trainer):
    """Flush and close TensorBoard writer cleanly at train end."""
    if WRITER:
        WRITER.flush()
        WRITER.close()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
    }
    if SummaryWriter
    else {}
)
