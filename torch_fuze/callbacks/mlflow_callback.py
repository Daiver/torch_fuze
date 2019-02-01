from ..trainer_state import TrainerState
from .abstract_callback import AbstractCallback

import os
import mlflow


class MLFlowCallback(AbstractCallback):
    # TODO: i need clear solution for category/metric
    def __init__(
            self,
            metrics_to_track: set=None,
            lowest_metrics_to_track: set=None,
            highest_metrics_to_track: set=None,
            files_to_save_at_every_batch: set=None
    ):
        super().__init__()
        self.metrics_to_track = metrics_to_track
        self.lowest_metrics_to_track = lowest_metrics_to_track if lowest_metrics_to_track is not None else set()
        self.highest_metrics_to_track = highest_metrics_to_track if highest_metrics_to_track is not None else set()
        self.files_to_save_at_every_batch = files_to_save_at_every_batch

        self.lowest_metric_values = {}
        self.highest_metric_values = {}

    def log_artifacts(self):
        if self.files_to_save_at_every_batch is None:
            return
        for file_name in self.files_to_save_at_every_batch:
            # mlflow.log_artifact(file_name, artifact_path=os.path.basename(file_name))
            mlflow.log_artifact(file_name)

    def log_metrics_from_state(self, state: TrainerState):
        for cat_name, metrics in state.metrics_per_category.items():
            for metric_name, metric_value in metrics.items():
                full_metric_name = "_".join((cat_name, metric_name))
                if self.metrics_to_track is not None and full_metric_name not in self.metrics_to_track:
                    continue
                mlflow.log_metric(full_metric_name, metric_value)

                if full_metric_name in self.lowest_metrics_to_track:
                    if full_metric_name not in self.lowest_metric_values:
                        self.lowest_metric_values[full_metric_name] = metric_value
                    else:
                        self.lowest_metric_values[full_metric_name] = min(
                            metric_value,
                            self.lowest_metric_values[full_metric_name])
                    mlflow.log_metric(full_metric_name + "_lowest", self.lowest_metric_values[full_metric_name])

                if full_metric_name in self.highest_metrics_to_track:
                    if full_metric_name not in self.highest_metric_values:
                        self.highest_metric_values[full_metric_name] = metric_value
                    else:
                        self.highest_metric_values[full_metric_name] = max(
                            metric_value,
                            self.highest_metric_values[full_metric_name])
                    mlflow.log_metric(full_metric_name + "_highest", self.highest_metric_values[full_metric_name])

    def on_epoch_end(self, state: TrainerState):
        self.log_metrics_from_state(state)
        self.log_artifacts()


