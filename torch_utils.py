""" Contains utility functions for PyTorch"""
# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)

import torch
import torch.optim as optim

# PrivacyBuckets slope for equal
SLOPE = 5.0


# Custom ceil implementation
class CeilDerivable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output.clone()
        )  # REMOVE CLONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Custom floor implementation
class FloorDerivable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output.clone()
        )  # REMOVE CLONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Custom equal function with gradient
class EqualDerivable(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, y_scalar):
        ctx.save_for_backward(x, y_scalar)
        return torch.eq(x, y_scalar).double()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        x, y_scalar = ctx.saved_tensors
        grad_x = grad_y_scalar = None

        if ctx.needs_input_grad[0]:
            x_rel = x - y_scalar
            lo = 1 / (1 + (x_rel * x_rel) / 0.01)  # Lorenz distribution
            grad_x = grad_output * lo

        if ctx.needs_input_grad[1]:
            raise NotImplementedError()
            grad_y_scalar = torch.mean(y_scalar)

        return grad_x, grad_y_scalar


customEqual = EqualDerivable.apply
customCeil = CeilDerivable.apply
customFloor = FloorDerivable.apply


# Custom <= with gradient
def customLessEqual(x, y):
    """
    derivable lessEqual. This function assumes that the input are integers
    (can still be floats, but close to integer values)
    """
    return torch.sigmoid(-SLOPE * (x - y - 0.5))


# Custom > with gradient
def customGreater(x, y):
    """ derivable greaterThan function """
    return torch.sigmoid(SLOPE * (x - y - 0.5))


def compute_privacy_loss_distribution(p_A_slice, p_B_slice, dist_events, dist_events_dual, privacy_loss, privacy_loss_dual, f, buckets_half):
    """Computes the privacy loss distribution

    Args:
        p_A_slice ([torch.DoubleTensor]): A
        p_B_slice ([torch.DoubleTensor]): B
        dist_events (torch.DoubleTensor): Distinguishing A/B
        dist_events_dual (torch.DoubleTensor): Distinguishing events B/A
        privacy_loss ([torch.DoubleTensor]): A/B
        privacy_loss_dual ([torch.DoubleTensor]): B/A
        f (float): Coarseness factor
        buckets_half (int): Buckets/2 to compute delta

    Returns:
        ([torch.DoubleTensor], [torch.DoubleTensor]): Privacy loss distribution and Privacy loss distribution dual
    """
    indices = torch.div(privacy_loss, torch.log(f)) + (2 * buckets_half + 1) // 2
    indices = customCeil(indices)

    indices_dual = torch.div(privacy_loss_dual, torch.log(f)) + (2 * buckets_half + 1) // 2
    indices_dual = customCeil(indices_dual)

    cond_first = customLessEqual(indices, 0)
    p_cond_first = torch.mul(cond_first, p_A_slice)

    cond_first_dual = customLessEqual(indices_dual, 0)
    p_cond_first_dual = torch.mul(cond_first_dual, p_B_slice)

    # create the PB lists
    list_of_tensors = [torch.sum(input=p_cond_first)]
    list_of_tensors_dual = [torch.sum(input=p_cond_first_dual)]

    # print("Constructing privacy loss distribution ..")
    for j in range(1, 2 * buckets_half + 1):
        j_tens = torch.tensor(j)
        cond = customEqual(indices, j_tens)
        p_cond = torch.mul(cond, p_A_slice)
        list_of_tensors.append(torch.sum(input=p_cond))

        cond_dual = customEqual(indices_dual, j_tens)
        p_cond_dual = torch.mul(cond_dual, p_B_slice)
        list_of_tensors_dual.append(torch.sum(input=p_cond_dual))

    cond_last = customGreater(indices, 2 * buckets_half)
    p_cond_last = torch.mul(cond_last, p_A_slice)
    list_of_tensors.append(torch.sum(input=p_cond_last) + dist_events)
    pb_distr = torch.stack(list_of_tensors, dim=0)

    cond_last_dual = customGreater(indices_dual, 2 * buckets_half)
    p_cond_last_dual = torch.mul(cond_last_dual, p_B_slice)
    list_of_tensors_dual.append(torch.sum(input=p_cond_last_dual) + dist_events_dual)
    pb_distr_dual = torch.stack(list_of_tensors_dual, dim=0)

    return pb_distr, pb_distr_dual


def get_gradient_printer(message):
    """Creates a gradient printer for torch. Beware -> can lead to huge amount of RAM usage!

    Args:
        message (String): Message to print
    """

    def p(grad):
        if grad.nelement() == 1:
            print(f"{message}: {grad}")
        else:
            print(f"{message}: shape: {grad.shape} max: {grad.max()} min: {grad.min()} mean: {grad.mean()} nans: {torch.isnan(grad).any()}")
        if torch.isnan(grad).any():
            raise Exception

    return p


def register_gradient_hook(variable, message):
    """Adds a hook to a variable and prints the gradient during the backward pass

    Args:
        variable (torch.Tensor): Tensor variable
        message (String): Message to print
    """
    # Always retain the gradient even if it is not a leaf tensor
    variable.retain_grad()
    # Register a gradient printer function
    variable.register_hook(get_gradient_printer(message))


def create_optimizer(args, parameters):
    """Creates a PyTorch optimizer that learns given parameters

    Args:
        args ([Arguments]): Arguments of the program, must contain 'learning_rate' and 'optimizer'
        parameters ([nn.Parameter]): Parameters to Optimitze

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    eta = args.learning_rate

    if args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=eta)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=eta)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=eta)
    elif args.optimizer == "RMSProp":
        optimizer = optim.RMSprop(parameters, lr=eta)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    return lr_decay, optimizer


def hook_fn(m, i, o):
    """Creates a hook

    Args:
        m : method
        i : Input gradient
        o : Output gradient
    """
    print(m)
    print("------------Input Grad------------")
    for grad in i:
        try:
            print(grad)
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad, grad.shape)
        except AttributeError:
            print("None found for Gradient")
    print("\n")


def add_gradient_noise(parameters, args, step, gamma=0.55):
    """Applies gradient noise as  described in https://ruder.io/optimizing-gradient-descent/index.html#gradientnoise
        Adapted after: Neelakantan et al. [31] add noise that follows a Gaussian distribution N(0,σ2t) to each gradient update
    Args:
        parameters ([torch.nn.Parameter]): Parameters we want to apply a noise
        args (Arguments): Program arguments
        step (int): epoch we are in
        gamma (float, optional): Decreasing factor. Defaults to 0.55.
    """
    with torch.no_grad():
        if args.gradient_noise:
            for _, p in parameters:
                std = torch.sqrt(torch.tensor(args.learning_rate / ((1 + step) ** gamma)).double())
                p.grad = p.grad + torch.normal(torch.zeros_like(p.grad), torch.zeros_like(p.grad) + std)


def alternate_lamda_delta_update(parameters, args, step):
    """Alternate lambda and delta update

    Args:
        parameters (torch.nn.Parameters): Parameters we want to learn
        args (Arguments): Program arguments
        step (int): epoch we are in
    """
    with torch.no_grad():
        if args.alternate_lamda_delta_update and args.method == "renyi_markov":
            for n, p in parameters:
                if "lam" in n and step % 2 != 0:  # Zero LAM in odd steps
                    p.grad = torch.zeros_like(p.grad)
                if "lam" not in n and step % 2 == 0:  # Zero everything else in even steps
                    p.grad = torch.zeros_like(p.grad)


def log_add(a, b):
    """Computes ln(p+q), where a is a=ln(p) and b=ln(q)

    Args:
        a (torch.DoubleTensor)
        b (torch.DoubleTensor)
    """
    if a > b:
        return torch.add(a, log1pexp(b - a))
    else:
        return torch.add(b, log1pexp(a - b))


def log1pexp(a):
    if a < torch.tensor(709.089565713, device=a.device, dtype=a.dtype):
        return torch.tensor(0.0, device=a.device, dtype=a.dtype)
    else:
        return torch.log1p(torch.exp(a))
