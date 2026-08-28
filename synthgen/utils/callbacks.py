import os
import tracemalloc
from pathlib import Path
from typing import Optional
import logging
import lightning.pytorch as pl


class TracemallocCallback(pl.Callback):
    """
    Tracks Python-level memory allocations via tracemalloc and reports growth.
    Notes:
      - This will NOT attribute native allocations (e.g., large NumPy buffers, PyTorch CPU allocator).
      - It IS useful for finding reference retention in Python (lists, dicts, images, strings, etc.).
    """

    def __init__(
        self,
        enabled: bool = True,
        n_frames: int = 25,
        report_top: int = 25,
        every_n_train_steps: Optional[int] = 200,
        every_n_epochs: Optional[int] = None,
        dump_dir: Optional[str] = None,
        key_type: str = "lineno",  # "lineno" or "filename"
        only_project: Optional[str] = None,  # e.g. "/path/to/prism"
    ):
        self.enabled = enabled
        self.n_frames = n_frames
        self.report_top = report_top
        self.every_n_train_steps = every_n_train_steps
        self.every_n_epochs = every_n_epochs
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.key_type = key_type
        self.only_project = Path(only_project).resolve() if only_project else None

        self._snap_prev = None
        self._snap_prev_tag = None
        # log info that is being tracked
        if self.enabled:
            logging.info(
                f"[TracemallocCallback] Enabled: n_frames={n_frames}, report_top={report_top}, "
                f"every_n_train_steps={every_n_train_steps}, every_n_epochs={every_n_epochs}, "
                f"key_type={key_type}, only_project={only_project}, dump_dir={dump_dir}"
            )

    def _should_report_step(self, trainer: "pl.Trainer") -> bool:
        if not self.every_n_train_steps:
            return False
        step = trainer.global_step
        return step > 0 and (step % self.every_n_train_steps == 0)

    def _should_report_epoch(self, trainer: "pl.Trainer") -> bool:
        if not self.every_n_epochs:
            return False
        epoch = trainer.current_epoch
        return epoch > 0 and (epoch % self.every_n_epochs == 0)

    def _filter_snapshot(self, snap: tracemalloc.Snapshot) -> tracemalloc.Snapshot:
        if self.only_project:
            # Keep only traces coming from your project directory
            return snap.filter_traces(
                [
                    tracemalloc.Filter(
                        inclusive=True, filename=str(self.only_project) + os.sep + "*"
                    )
                ]
            )
        return snap

    def _report(self, trainer: "pl.Trainer", tag: str) -> None:
        snap = tracemalloc.take_snapshot()
        snap = self._filter_snapshot(snap)

        if self._snap_prev is None:
            self._snap_prev = snap
            self._snap_prev_tag = tag
            return

        stats = snap.compare_to(self._snap_prev, self.key_type)

        header = f"[tracemalloc] growth since '{self._snap_prev_tag}' -> '{tag}'"
        # Lightning-safe printing: use rank_zero_only via trainer if you want; keep simple here
        print("\n" + header)
        for i, stat in enumerate(stats[: self.report_top], start=1):
            print(f"{i:02d}. {stat}")

        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            out = self.dump_dir / f"tracemalloc_{trainer.global_step:08d}_{tag}.txt"
            out.write_text(
                header
                + "\n"
                + "\n".join(str(s) for s in stats[: self.report_top])
                + "\n"
            )

        self._snap_prev = snap
        self._snap_prev_tag = tag

    # ---- Lightning hooks ----

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.enabled:
            return
        tracemalloc.start(self.n_frames)
        self._snap_prev = self._filter_snapshot(tracemalloc.take_snapshot())
        self._snap_prev_tag = "fit_start"

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not self.enabled:
            return
        if self._should_report_step(trainer):
            self._report(trainer, f"train_step_{trainer.global_step}")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.enabled:
            return
        if self._should_report_epoch(trainer) or (
            self.every_n_epochs is None and self.every_n_train_steps is None
        ):
            self._report(trainer, f"train_epoch_end_{trainer.current_epoch}")

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.enabled:
            return
        # final report
        self._report(trainer, "fit_end")
        tracemalloc.stop()
