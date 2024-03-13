import warnings
from torch.utils.data import IterableDataset


def check_batch_params(args, num_devices):
    if args.global_batch_size:
        args.batch_size = args.global_batch_size // num_devices
        warnings.warn('Per gpu batch size set at ', args.batch_size)
    elif args.batch_size:
        args.global_batch_size = args.batch_size * num_devices
        warnings.warn(
            f'Global batch size set to {args.global_batch_size}. Using batch size * accelerator.num_processes (num devices)')
    elif not (args.global_batch_size or args.batch_size):
        warnings.warn(
            f'Global batch size and batch size arguments not set, using deepspeed config for batch size for MLM')
        if args.clip:
            raise AssertionError('set either global_batch_size or batch_size argument for CLIP')
    return args


def check_iter_params(args, trainloader):
    if not ( args.epochs or args.total_steps):
        raise AssertionError('set either epochs or iterations argument for training')
    if not args.total_steps:
        if args.mlm:
            args.total_steps = len(trainloader) * args.epochs
        elif args.clip:
            args.total_steps = args.train_num_batches * args.epochs
            warnings.warn(f'total steps {args.total_steps} as trainloader is IterableDataset with no len')
    return args

def check_train_selection(args):
    if not (args.mlm or args.clip):
        raise AssertionError('set either mlm or clip argument')

def check_grad_acc_steps(args, accelerator):
    if args.gradient_accumulation_steps:
        accelerator.deepspeed_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    return args

