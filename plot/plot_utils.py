import numpy as np
import matplotlib.pyplot as plt
from plot.plot_minigrid import plot_grid
import ipdb

def draw_act_dist(env, traj_info_dict, idx_lists, state_ini=None, name='Action_FourRoom', save_img=True):
    height, width = env.env.grid.height, env.env.grid.width
    vis_grid = env.env.grid.grid

    trajPos = traj_info_dict['pos']
    trajDirect = traj_info_dict['dir']
    trajAct = traj_info_dict['act']
    trajSkill = traj_info_dict['skill']

    for group_id, idx_list in enumerate(idx_lists):
        move_array = np.zeros((height, width, 4)) #
        skill_array = np.zeros((height, width, 8))
        for idx in idx_list:
            pos_traj = trajPos[idx]
            dir_traj = trajDirect[idx]
            act_traj = trajAct[idx]
            skill_traj = trajSkill[idx]
            ipdb.set_trace()
            for move in (act_traj==2).nonzero()[0]:
                pos = pos_traj[move] # (column, row)
                dir = dir_traj[move]
                move_array[pos[1], pos[0], int(dir)] += 1
                skill = skill_traj[move]
                skill_array[pos[1], pos[0], int(skill)] += 1

        if state_ini is not None:
            map_grid = state_ini.grid.grid # list of grid items, with length (height * width)
        else:
            map_grid = vis_grid #

        fig = plot_grid(height, width, map_grid, move_array, skill_array,)

        if save_img:
            plt.imsave('imgs_new/{}_group{}.jpg'.format(name, group_id),fig)


def draw_skill_traj(env, traj_info_dict, state_ini=None, name='Action_FourRoom'):
    height, width = env.env.grid.height, env.env.grid.width
    vis_grid = env.env.grid.grid

    pos_traj = traj_info_dict['pos']
    dir_traj = traj_info_dict['dir']
    act_traj = traj_info_dict['act']
    skill_traj = traj_info_dict['skill']

    dir_array = np.zeros((height, width, 4))
    skill_array = np.zeros((height, width, 8))
    for step in range(len(skill_traj)):
        pos = pos_traj[step] # (column, row)
        dir = dir_traj[step]
        dir_array[pos[1], pos[0], int(dir)] += 1
        skill = skill_traj[step]
        skill_array[pos[1], pos[0], int(skill)] += 1

    if state_ini is not None:
        map_grid = state_ini.grid.grid # list of grid items, with length (height * width)
    else:
        map_grid = vis_grid #

    fig = plot_grid(height, width, map_grid, dir_array, skill_array, filter_rare=False)

    plt.imsave('imgs_new/{}.jpg'.format(name),fig)