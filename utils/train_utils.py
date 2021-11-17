import torch
import torch.optim as optim
from torch.optim import Adam


class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.state_dict()}, path)


def generator_step(net_d, out2, net_loss, optimizer):
    set_requires_grad(net_d, False)
    d_fake = net_d(out2[:, 0:2048, :])
    errG_loss_batch = torch.mean((d_fake - 1) ** 2)
    total_gen_loss_batch = errG_loss_batch + net_loss * 200
    total_gen_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda(), retain_graph=True, )
    optimizer.step()
    return d_fake


def discriminator_step(net_d, gt, d_fake, optimizer_d):
    set_requires_grad(net_d, True)
    d_real = net_d(gt[:, 0:2048, :])
    d_loss_fake = torch.mean(d_fake ** 2)
    d_loss_real = torch.mean((d_real - 1) ** 2)
    errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
    total_dis_loss_batch = errD_loss_batch
    total_dis_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda())
    optimizer_d.step()

def get_optimizer(net, args, lr):
    optimizer = getattr(optim, args.optimizer)
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))

    # split out the optimizers for extra modules
    optim_parts = getattr(args, 'optim_parts', None)
    if isinstance(optim_parts, list):
        optim_parts = {part:args.lr for part in optim_parts}
    if optim_parts is not None:
        main_params = []
        sub_params = {}
        sub_lrs = []
        for name, param in net.module.named_parameters():
            is_main = True
            for part in optim_parts:
                if part in name:
                    if part in sub_params:
                        sub_params[part].append(param)
                    else:
                        sub_params[part] = [param]
                        sub_lrs.append(optim_parts[part])
                    is_main = False
                    break
            if is_main:
                main_params.append(param)
        print(sub_params.keys())
        optimizer = optimizer(main_params, lr=lr, weight_decay=args.weight_decay, betas=betas)
        sub_optimizers = [
            Adam(params, lr=lr, weight_decay=args.weight_decay, betas=betas) for params in sub_params.values()
        ]
    else:
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
        sub_optimizers = []
        sub_lrs = []

    return optimizer, sub_optimizers, sub_lrs