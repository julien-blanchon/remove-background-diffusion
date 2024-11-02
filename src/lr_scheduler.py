import torch


class IterExponential:
    def __init__(
        self, total_iter_length: int, final_ratio: float, warmup_steps: int = 0
    ) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter: int) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = torch.exp(
                actual_iter / self.effective_length * torch.log(self.final_ratio)
            )
        return alpha
