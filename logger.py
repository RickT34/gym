from tensorboardX import SummaryWriter
import numpy as np
import datetime

class Logger:
    def __init__(self, label:str):
        log_dir = f'logs/{label}'
        self._log_dir = log_dir
        print('logging outputs to ', log_dir)
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        self._step = 0

    def log_scalar(self, name, val):
        self._summ_writer.add_scalar('{}'.format(name), val, self._step)

    def log_video(self, name, render_frames, fps=30):
        video_frames = np.stack(render_frames, axis=0).transpose(0, 3, 1, 2)
        video_frames = np.expand_dims(video_frames, axis=0)
        self._summ_writer.add_video('{}'.format(name), video_frames, self._step, fps=fps)

    def commit(self):
        self._summ_writer.flush()
        self._step += 1



