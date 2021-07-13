# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
import torch


class SymmetricNoise(torch.nn.Module):
    """Symmetric Differential Privacy Preparer"""

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()
        self.element_size = element_size
        self.Delta = int(element_size / (2 * range_begin))
        self.device = device

    def forward(self, p):
        A = torch.zeros(int(self.element_size + self.Delta), device=self.device)
        B = torch.zeros(int(self.element_size + self.Delta), device=self.device)

        A[: self.element_size] = p
        B[-self.element_size :] = p

        p_A_slice = p[: self.element_size - self.Delta]
        p_B_slice = p[self.Delta :]

        dist_events = torch.sum(p[self.element_size - self.Delta :])
        dist_events_dual = torch.sum(p[: self.Delta])

        return p_A_slice, p_B_slice, dist_events, dist_events_dual, A, B


class MixtureNoise(torch.nn.Module):
    """Mixture Differential Privacy Preparer"""

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()
        self.element_size = element_size
        self.Delta = int(element_size / (2 * range_begin))
        self.q = torch.tensor(args.mixture_q, device=device)
        self.device = device

    def forward(self, p):
        A = torch.zeros(int(self.element_size + self.Delta), device=self.device)
        B = torch.zeros(int(self.element_size + self.Delta), device=self.device)

        A[: self.element_size] = p

        B[: self.element_size] = (1.0 - self.q) * p
        B[-self.element_size :] += self.q * p

        B /= torch.sum(B)

        p_A_slice = A[: self.element_size]
        p_B_slice = B[: self.element_size]
        dist_events = torch.tensor(0.0, device=self.device)  # (A[B==0]) A/B
        dist_events_dual = torch.sum(B[-self.Delta :])  # (B[A==0]) B/A
        return p_A_slice, p_B_slice, dist_events, dist_events_dual, A, B
