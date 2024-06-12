from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    def __init__(self, render_freq):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True
