import torch
import numpy as np

import torch_utils
import utils


def convolution_full_1d(x_tens, y_tens):
    assert len(x_tens.shape) == 1
    assert len(y_tens.shape) == 1
    # L_out = x_tens.shape[0] + y_tens.shape[0] - 1

    assert x_tens.shape[0] % 2 == 1
    assert x_tens.shape[0] == y_tens.shape[0]

    # using numpy 'same' padding definition.
    # using the formula given from torch doc:
    #   L_out​ = [(L_in​ + 2 * padding - dilation * (kernel_size − 1) − 1​ ) / stride ] + 1
    # with stride = delation = 1 leads to
    #   padding = (L_out - L_in + kernel_size - 1) / 2
    padding = y_tens.shape[0] - 1

    x_tens = x_tens.reshape((1, 1, -1))
    y_tens = y_tens.reshape((1, 1, -1))

    y_tens_rev = y_tens.flip(dims=(2,))

    conved = torch.conv1d(x_tens, y_tens_rev, padding=padding)

    assert conved.shape[2] == x_tens.shape[2] + y_tens.shape[2] - 1

    return conved.reshape(-1)


class ComputePrivacyBucketsDelta(torch.nn.Module):
    def __init__(self, args, delta_type, device):
        super().__init__()
        self.fastmode = args.fastmode
        self.eps = args.eps

        assert delta_type in ["ADP", "PDP"]
        self.delta_type = delta_type

        # we only support integer exponentials of 2 as number_of_compositions
        self.number_of_self_convolutions = int(np.log2(args.number_of_compositions))
        assert (
            2 ** int(self.number_of_self_convolutions) == args.number_of_compositions
        ), f"number_of_compositions ({args.number_of_compositions}) is not an integer exponential of 2"

        # shall be divisible by 2
        assert args.buckets_half % 2 == 0
        buckets_half_numpy = np.int64(args.buckets_half)
        self.buckets_half = torch.tensor(args.buckets_half, device=device)
        self.vector_size = 2 * self.buckets_half.clone().detach() + 2
        self.extended_size = 4 * self.buckets_half.clone().detach() + 1

        f = np.float64(args.factor)
        assert f > 1
        # adapt factor to eps.
        f = utils.get_good_factor(initial_factor=f, eps=args.eps, buckets_half=buckets_half_numpy)
        self.f = torch.tensor(f, device=device)

        # prepare g_arr

        # k is the first (smallest) index with eps < np.log(f**k)
        k = int(np.floor(self.eps / np.log(f)))
        g_arr = np.zeros(2 * buckets_half_numpy + 2)

        if self.delta_type == "ADP":

            def _g_func(idx):
                return 1 - f ** -idx

            g_arr[buckets_half_numpy + k + 1 : -1] = _g_func(np.arange(1, buckets_half_numpy - k + 1))
            g_arr[-1] = 1

        elif self.delta_type == "PDP":
            g_arr[buckets_half_numpy + k + 1 : -1] = np.ones(buckets_half_numpy - k)
            g_arr[-1] = 1

        else:
            raise ValueError(f"delta_type '{self.delta_type}' not supported ")

        self.g_tensor = torch.tensor(g_arr, device=device)

        # preapare constant returns

        # for simplicity, we included the dist events in the infty bucket
        # and thereby it is already incroporated in delta.
        self.dist_events_comp = torch.tensor(0, device=device)
        self.dist_events_comp_dual = torch.tensor(0, device=device)

    def forward(self, p_A_slice, p_B_slice, dist_events, dist_events_dual, step):

        # Calculate the discretisation

        log_A = torch.log(p_A_slice)
        log_B = torch.log(p_B_slice)

        privacy_loss = log_A - log_B
        privacy_loss_dual = -privacy_loss  # pylint: disable=invalid-unary-operand-type

        pb_distr, pb_distr_dual = torch_utils.compute_privacy_loss_distribution(
            p_A_slice, p_B_slice, dist_events, dist_events_dual, privacy_loss, privacy_loss_dual, self.f, self.buckets_half
        )

        # proto = tf.zeros([1], dtype=tf.float64)
        proto = torch.tensor(0)
        for j in range(self.number_of_self_convolutions):
            # assert False, "we should not reach here currently"

            # conv_j
            # pb_distr_inner = tf.slice(pb_distr, [0], [self.vector_size - 1], name="pb_distr_inner")
            pb_distr_inner = pb_distr[: self.vector_size - 1]
            B_extended = convolution_full_1d(pb_distr_inner, pb_distr_inner)
            inf_bucket = pb_distr[-1]
            inf_bucket = proto + inf_bucket + inf_bucket - inf_bucket * inf_bucket
            inf_bucket = inf_bucket + torch.sum(input=B_extended[self.extended_size - self.buckets_half : self.extended_size])
            self.minus_n_bucket = proto + torch.sum(input=B_extended[: self.buckets_half + 1])
            self.mid_buckets = B_extended[self.buckets_half + 1 : self.buckets_half + 1 + self.vector_size - 2]
            pb_distr = torch.cat(tensors=(self.minus_n_bucket.reshape((1,)), self.mid_buckets, inf_bucket.reshape((1,))), dim=0)

            # conv_j dual
            # pb_distr_dual_inner = tf.slice(pb_distr_dual, [0], [self.vector_size - 1], name="pb_distr_inner_dual")
            pb_distr_dual_inner = pb_distr_dual[: self.vector_size - 1]
            B_extended_dual = convolution_full_1d(pb_distr_dual_inner, pb_distr_dual_inner)
            inf_bucket_dual = pb_distr_dual[-1]
            inf_bucket_dual = proto + inf_bucket_dual + inf_bucket_dual - inf_bucket_dual * inf_bucket_dual
            inf_bucket_dual = inf_bucket_dual + torch.sum(input=B_extended_dual[self.extended_size - self.buckets_half : self.extended_size])
            minus_n_bucket_dual = proto + torch.sum(input=B_extended_dual[: self.buckets_half + 1])
            mid_buckets_dual = B_extended_dual[self.buckets_half + 1 : self.buckets_half + 1 + self.vector_size - 2]
            pb_distr_dual = torch.cat(tensors=(minus_n_bucket_dual.reshape((1,)), mid_buckets_dual, inf_bucket_dual.reshape((1,))), dim=0)
            # del inf_bucket
            # del minus_n_bucket
            # del B

            # pb_distr = tf.identity(pb_distr, name="pb_distr_after_conv")
            # pb_distr_dual = tf.identity(pb_distr_dual, name="pb_distr_dual_after_conv")

        # computing delta

        # print("Computing the delta ..")

        pb_distr_times_g = torch.mul(pb_distr, self.g_tensor)
        pb_distr_dual_times_g = torch.mul(pb_distr_dual, self.g_tensor)
        pb_delta = torch.sum(input=pb_distr_times_g)
        pb_delta_dual = torch.sum(input=pb_distr_dual_times_g)

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
