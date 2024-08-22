# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import mmengine.dist as dist
import rich.progress as progress
from rich.live import Live

disable_progress_bar = False
global_progress = progress.Progress(
    '{task.description}',
    progress.BarColumn(),
    progress.TaskProgressColumn(show_speed=True),
    progress.TimeRemainingColumn(),
)
global_live = Live(global_progress, refresh_per_second=10)


def track(sequence, description: str = '', total: Optional[float] = None):
    if disable_progress_bar:
        yield from sequence
    else:
        global_live.start()
        task_id = global_progress.add_task(description, total=total)
        task = global_progress._tasks[task_id]
        try:
            yield from global_progress.track(sequence, task_id=task_id)
        finally:
            if task.total is None:
                global_progress.update(task_id, total=task.completed)
            if all(task.finished for task in global_progress.tasks):
                global_live.stop()
                for task_id in global_progress.task_ids:
                    global_progress.remove_task(task_id)


def track_on_main_process(sequence, description='', total=None):
    if not dist.is_main_process() or disable_progress_bar:
        yield from sequence
    else:
        yield from track(sequence, total=total, description=description)
