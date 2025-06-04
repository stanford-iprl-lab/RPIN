import os
import hickle
import argparse
import numpy as np
import random
from virtualtools import get_feasible_actions, simulate_single_puzzle

output_size = 128
mask_size = 21

def arg_parse():
    parser = argparse.ArgumentParser(description='Generate virtual tools data for RPIN')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--set', type=str, required=True)
    parser.add_argument('--puzzle_name', type=str, required=True)
    parser.add_argument('--data_folder_name', type=str, required=True)
    parser.add_argument('--with_variations', type=str, required=True, choices=['True', 'False'])
    parser.add_argument('--tool', type=str, default='Best')
    parser.add_argument('--num_actions', type=int, default=6)
    parser.add_argument('--balance', type=float, default=0.6)
    parser.add_argument('--no_action', type=float, default=0.05)
    
    return parser.parse_args()


def gen_single_task(set, name, tool, data_folder_name, data_path, num_actions=10, balance=0.5, with_variations=True, no_action_runs=0.05): 
    random.seed(0)

    tool_list = []
    if tool == 'Best':
        tool_list = ['Best']
    elif tool == 'All':
        tool_list = ['obj1', 'obj2', 'obj3']
    else:
        if tool in ['obj1', 'obj2', 'obj3']:
            tool_list = [tool]
        else:
            print('Invalid tool')
            return
    no_action_runs = int(no_action_runs * num_actions // len(tool_list))
    extra_actions_per_tool = int(num_actions*0.1 // len(tool_list)) # to compensate for actions that the simulator fails to generate for unknown reasons
    actions_per_tool = int((num_actions-no_action_runs) // len(tool_list))
    
    path = os.path.dirname(os.path.abspath(__file__))
    path = '/'.join(path.split('/')[:-2])
    path = os.path.join(path, "virtual-tools-backbone", "virtualtools", "trials", set)

    if with_variations:
        puzzles = os.listdir(os.path.join(path, name))
    else:
        puzzles = [name + '.json']

    for variation in puzzles:
        if '.json' not in variation:
            continue
        actions_offset = 0
        variation_name = variation.split('.')[0]

        for tool in tool_list:
            actions = get_feasible_actions(set, name, variation, actions_per_tool + extra_actions_per_tool, tool=tool, balance=balance, with_variations=with_variations)
            # add no action runs as None to the actions list
            actions += [None] * no_action_runs
            images, labels, boxes, masks, actions = simulate_single_puzzle(set, name, variation, actions, size=actions_per_tool+no_action_runs, tool=tool, output_size=output_size, mask_size=mask_size, with_variations=with_variations)
            actions = actions[:actions_per_tool+no_action_runs]
            im_save_root = f'{data_path}/images/{data_folder_name}/{variation_name}'
            fim_save_root = f'{data_path}/full/{data_folder_name}/{variation_name}'
            bm_save_root = f'{data_path}/labels/{data_folder_name}/{variation_name}'
            os.makedirs(im_save_root, exist_ok=True)
            os.makedirs(fim_save_root, exist_ok=True)
            os.makedirs(bm_save_root, exist_ok=True)

            print("feasible actions: ", len(actions))

            for act_id in range(len(actions)):
                action_id = act_id + actions_offset
                image = np.array(images[act_id])
                box = np.array(boxes[act_id])
                mask = np.array(masks[act_id])
                np.save(f'{im_save_root}/{action_id:04d}.npy', images[act_id][0])
                hickle.dump(image, f'{fim_save_root}/{action_id:04d}_image.hkl', mode='w', compression='gzip')
                hickle.dump(labels[act_id], f'{bm_save_root}/{action_id:04d}_label.hkl', mode='w', compression='gzip')
                hickle.dump(box, f'{bm_save_root}/{action_id:04d}_boxes.hkl', mode='w', compression='gzip')
                hickle.dump(mask, f'{bm_save_root}/{action_id:04d}_masks.hkl', mode='w', compression='gzip')
            actions_offset += len(actions)

def _get_puzzle_names(set, data_path):
    filenames = os.listdir(f'{data_path}/{set}')
    puzzle_names = [f.split('.')[0] for f in filenames]
    return puzzle_names


if __name__ == '__main__':
    random.seed(0)
    args = arg_parse()
    
    data_dir = args.data
    os.makedirs(data_dir, exist_ok=True)

    gen_single_task(args.set, args.puzzle_name, args.tool, args.data_folder_name, args.output, args.num_actions, args.balance, True if args.with_variations == 'True' else False, args.no_action)


# python dynamics/tools/gen_virtual_tools.py --data ../virtual-tools-backbone/virtualtools/trials/ --output . --set Original --puzzle_name Basic --balance 0.5 --num_actions 6 --tool All --data_folder_name Basic_06_All_Test --with_variations False --no_action 0.05