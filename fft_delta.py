# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# according to
#      "Computing Differential Privacy Guarantees for Heterogeneous Compositions Using FFT"
#      (Antti Koskela, Antti Honkela, https://arxiv.org/pdf/2102.12412.pdf)

import torch
import torch.fft
import numpy as np

import torch_utils
import utils


class ComputeFFTDelta(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.fastmode = args.fastmode
        self.eps = args.eps
        self.number_of_compositions = args.number_of_compositions
        self.device = device

        # shall be divisible by 2
        assert args.buckets_half % 2 == 0
        buckets_half_numpy = np.int64(args.buckets_half)
        self.buckets_half = torch.tensor(args.buckets_half, device=device)
        self.vector_size = 2 * torch.tensor(args.buckets_half, device=device) + 2
        f = np.float64(args.factor)
        assert f > 1
        # adapt factor to eps.
        f = utils.get_good_factor(initial_factor=f, eps=args.eps, buckets_half=buckets_half_numpy)
        self.f = torch.tensor(f, device=device)

        self.L = torch.log(self.f) * 2 * self.buckets_half  # torch.tensor(args.range_begin, device=device)  #

        self._lambda = self.L / 2
        self.n = 2 * self.buckets_half
        self.delta_x = 2 * self.L / self.n
        self.discretization = torch.linspace(-self.L, self.L - self.delta_x, self.n, device=device)
        self.dist_events_comp = torch.tensor(0, device=device)
        self.dist_events_comp_dual = torch.tensor(0, device=device)
        self.min_index = torch.floor((self.n * (self.L + self.eps) / (2 * self.L))).int()
        self.error_factor = torch.exp(-self._lambda * self.L) / (1 - torch.exp(-2 * self._lambda * self.L))

    def forward(self, p_A_slice, p_B_slice, dist_events, dist_events_dual, step):
        log_A = torch.log(p_A_slice)
        log_B = torch.log(p_B_slice)

        privacy_loss = log_A - log_B
        privacy_loss_dual = -privacy_loss

        pb_delta = self.computeFFTDelta(privacy_loss, p_A_slice, dist_events, self.computeError(p_A_slice, p_B_slice))
        pb_delta_dual = self.computeFFTDelta(privacy_loss_dual, p_B_slice, dist_events, self.computeError(p_B_slice, p_A_slice))

        if self.fastmode:
            debug_tensors = {}
        else:
            debug_tensors = {
                "privacy_loss": privacy_loss,
                "privacy_loss_dual": privacy_loss_dual,
            }

        # self.dist_events_comp and self.dist_events_comp_dual are always 0 and they are only returned
        # so that the function returns the same as renyi markov delta
        return pb_delta, pb_delta_dual, self.dist_events_comp, self.dist_events_comp_dual, torch.add(pb_delta, self.dist_events_comp), debug_tensors

    def computeError(self, p_A_slice, p_B_slice):
        """Computes the error according to Theorem 10 of Koskella et al. "Tight Differential Privacy for Discrete-Valued Mechanismsand for the Subsampled Gaussian Mechanism Using FFT"

        [1] https://arxiv.org/pdf/2006.07134.pdf

        Args:
            p_A_slice (torch.DoubleTensor): p_A
            p_B_slice (torch.DoubleTensor): p_B

        Returns:
            torch.DoubleTensor: error
        """
        privacy_loss = torch.log(p_A_slice) - torch.log(p_B_slice)

        alpha_plus = torch.logsumexp(privacy_loss * self._lambda + torch.log(p_A_slice), 0)
        alpha_minus = torch.logsumexp(-privacy_loss * self._lambda + torch.log(p_B_slice), 0)
        exp_alpha_plus = torch.exp(alpha_plus)
        exp_alpha_minus = torch.exp(alpha_minus)

        T1 = (2 * torch.exp((self.number_of_compositions + 1) * alpha_plus) - torch.exp(self.number_of_compositions * alpha_plus) - exp_alpha_plus) / (exp_alpha_plus - 1)
        T2 = (torch.exp((self.number_of_compositions + 1) * alpha_minus) - exp_alpha_minus) / (exp_alpha_minus - 1)
        error = (T1 + T2) * self.error_factor

        return error

    def computeFFTDelta(self, privacy_loss, p_slice, dist_events, error):
        """Computes the delta according to Koskella et al. "Tight Differential Privacy for Discrete-Valued Mechanismsand for the Subsampled Gaussian Mechanism Using FFT"

        [1] https://arxiv.org/pdf/2006.07134.pdf

        Args:
            privacy_loss ([torch.DoubleTensor]): PLD
            p_slice ([torch.DoubleTensor]): slice of the distribution
            dist_events (torch.DoubleTensor): Distinguishing events
            error (torch.DoubleTensor): Error

        Returns:
            [torch.DoubleTensor]: delta
        """        
        """
        """
        pb_distr = torch.zeros(self.n, device=self.device)

        indices = torch_utils.customCeil((self.L + privacy_loss) / self.delta_x)
        for i in range(0, self.n):
            ii = torch.tensor(i, device=self.device)
            cond = torch_utils.customEqual(indices, ii)
            p_cond = torch.mul(cond, p_slice)
            pb_distr[i] += torch.sum(p_cond)

        # Flip with D = [0 I;I 0]
        n_2 = self.n // 2
        temp = pb_distr[n_2:]
        pb_distr[n_2:] = pb_distr[:n_2]
        pb_distr[:n_2] = temp

        # Compute the FFT
        fft_result = torch.fft.fft(pb_distr)
        tmp = fft_result.clone().detach()

        # If not -1, we would be off by one
        for _ in range(self.number_of_compositions - 1):
            fft_result *= tmp

        # Compute the IFFT
        c_k = torch.fft.ifft(fft_result)

        # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
        temp = c_k[n_2:]
        c_k[n_2:] = c_k[:n_2]
        c_k[:n_2] = temp
        c_k = c_k

        integrand = c_k * (1 - torch.exp(self.eps - self.discretization))
        dist_events_comp = 1 - torch.pow(1 - dist_events, self.number_of_compositions)
        delta = dist_events_comp + torch.sum(integrand[self.min_index + 1 :]).real + error
        print(delta, error)
        return delta

        # torch.autograd.set_detect_anomaly(True)
        # infty_bucket = pb_distr[-1]
        # pb_distr = pb_distr[1 : self.vector_size - 1]

        # # Flip with D = [0 I;I 0]
        # temp = pb_distr[self.buckets_half:]
        # pb_distr[self.buckets_half:] = pb_distr[: self.buckets_half]
        # pb_distr[: self.buckets_half] = temp

        # # Compute FFT
        # fft_result = torch.fft.fft(pb_distr * self.delta_x)
        # tmp = fft_result.clone().detach()

        # # If not -1, we would be off by one
        # for _ in range(self.number_of_compositions - 1):
        #     fft_result *= tmp

        # # Compute the inverse FFT
        # c_k = torch.fft.ifft(fft_result / self.delta_x).real

        # # Flip back to normal with D = [0 I;I 0]
        # temp = c_k[self.buckets_half:]
        # c_k[self.buckets_half:] = c_k[: self.buckets_half]
        # c_k[: self.buckets_half] = temp

        # delta = (1 - torch.pow(1 - infty_bucket, self.number_of_compositions)) + self.delta_x * torch.sum(self.factors * c_k[self.l_epsilon :])
        # return delta
