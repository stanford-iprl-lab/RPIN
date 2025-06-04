import os
import torch
import random
import argparse
import numpy as np
import json

from pprint import pprint
from torch.utils.data import DataLoader

from rpin.models import rpcin, rpcin_vae
from rpin.datasets import *
from rpin.utils.config import _C as C
from rpin.evaluator_plan import PlannerPHYRE
from rpin.evaluator_pred import PredEvaluator


def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--predictor-init', type=str, help='', default=None)
    parser.add_argument('--predictor-arch', type=str, default=None)
    parser.add_argument('--plot-image', type=int, default=0, help='how many images are plotted')
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--start_id', default=0, type=int)
    parser.add_argument('--end_id', default=25, type=int)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = arg_parse()
    pprint(vars(args))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available() and args.device != 'cpu':
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = torch.cuda.device_count()
        print('Use {} GPUs'.format(num_gpus))
    else:
        num_gpus = 1
        print('Use CPU')
        assert NotImplementedError

    # --- setup config files
    C.merge_from_file(args.cfg) 
    # C.RPIN.MAX_NUM_OBJS = 4
    C.freeze()

    cache_name = ''
    if args.predictor_init:
        cache_name += args.predictor_init.split('/')[-2]
    cache_name += '/figures'
    output_dir = os.path.join(C.OUTPUT_DIR, cache_name)

    print('output_dir:', output_dir)

    if args.eval == 'plan':
        assert 'reasoning' in C.DATA_ROOT
        assert num_gpus == 1, 'multi-gpu support is not avaialbe for planning tasks'
        model = eval(args.predictor_arch + '.Net')()
        model.to(torch.device(args.device))
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        # load prediction model
        cp = torch.load(args.predictor_init, map_location=args.device + ':0')
        model.load_state_dict(cp['model'])

        tester = PlannerPHYRE(
            device=torch.device(args.device),
            num_gpus=1,
            model=model,
            output_dir=output_dir,
        )
        tester.test(args.start_id, args.end_id)
        return

    # --- setup data loader
    print('initialize dataset')
    split_name = 'test'
    val_set = eval(f'{C.DATASET_ABS}')(data_root=C.DATA_ROOT, split=split_name, image_ext=C.RPIN.IMAGE_EXT, id_filter=(args.start_id,args.end_id))
    batch_size = 1 if C.RPIN.VAE else C.SOLVER.BATCH_SIZE * num_gpus
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=12)

    model = eval(args.predictor_arch + '.Net')()
    model.to(torch.device(args.device))
    if args.device != 'cpu':
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.gpus.count(',') + 1))
        )
    cp = torch.load(args.predictor_init, map_location=f'{args.device if args.device == "cpu" else "cuda:0"}')
    model_dict = {}
    for k in cp['model']:
        model_dict[k.replace('module.', '')] = cp['model'][k]
    model.load_state_dict(model_dict)
    tester = PredEvaluator(
        device=torch.device(args.device),
        val_loader=val_loader,
        num_gpus=num_gpus,
        model=model,
        num_plot_image=args.plot_image,
        output_dir=output_dir,
    )
    losses = tester.test()
    dir_path = os.path.join('/'.join(output_dir.split('/')[:-1]),"losses")
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, f'losses_{args.start_id}_{args.end_id}.json'), 'w') as fout:
        json.dump(losses, fout)
    # mean bounding box position loss for step < test size
    # m_1, m_2 mask loss
    # p_1, p_2 bounding box position loss
    # s_1, s_2 bounding box size loss
    # _1 step < train size
    # _2 step > train size


if __name__ == '__main__':
    main()
