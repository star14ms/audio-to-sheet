import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing_extensions import override
from typing import Any, Optional
from rich.progress import TaskID, TextColumn
from pytorch_lightning.callbacks.progress.rich_progress import BatchesProcessedColumn, CustomBarColumn, CustomTimeColumn, ProcessingSpeedColumn


class RichProgressBarCustom(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.theme.progress_bar = 'bar.complete'
        # self.theme.progress_bar_finished = 'bar.finished'
        # self.theme.progress_bar_pulse = 'bar.pulse'

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"Iter {self.trainer.global_step}"
        if self._trainer.fit_loop.max_batches is not None:
            train_description += f"/{self._trainer.fit_loop.max_batches}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description
    
    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)
        self.train_progress_bar0_id = self.progress.add_task(description='Epoch {}/{}'.format(self._trainer.current_epoch+1, self._trainer.max_epochs), total=self._trainer.max_epochs)

    def _update(self, progress_bar_id: Optional["TaskID"], current: int, visible: bool = True, description: str = '') -> None:
        if self.progress is not None and self.is_enabled:
            assert progress_bar_id is not None
            total = self.progress.tasks[progress_bar_id].total
            assert total is not None
            if not self._should_update(current, total):
                return
              
            if description == '':
                description = self.progress.tasks[progress_bar_id].description

            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(progress_bar_id, advance=advance, visible=visible, description=description)
            self.refresh()

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        train_description = "Iter {}/{}".format(batch_idx + 1, self._trainer.fit_loop.max_batches)
        self._update(self.train_progress_bar_id, batch_idx + 1, description=train_description)
        self._update_metrics(trainer, pl_module)
        self.refresh()

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.progress.update(self.train_progress_bar0_id, advance=1)
        if self._trainer.current_epoch + 1 == self._trainer.max_epochs:
            self.progress.stop()
        self._update_metrics(trainer, pl_module)

    def configure_columns(self, trainer: "pl.Trainer") -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]
