import ipdb
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows
import matplotlib.font_manager as fm
import ipdb
import math

TILE_PIXELS = 32


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0]//factor, factor, img.shape[1]//factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn

def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout

def render_tile(cell, tile_size=16, subdivs=3, move=None, skill=None, filter_rare=True, skill_confidence=False):
    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    if filter_rare: # set threshold to filter out infrequent actions
        thresh = 3
    else:
        thresh = 1

    if cell is not None:
        cell.render(img)
    elif move.sum() >= thresh: # draw arrow
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        agent_dir = move.argmax()
        if skill_confidence:
            agent_confidence = skill.max()/skill.sum()+0.00001
        else:
            agent_confidence = move.max()/move.sum()+0.000001

        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        if skill is not None:
            skill_id = int(skill.argmax())
            color_map = [[255, 0,0], [255, 128,0], [255, 255, 0], [0,255,0],[51, 255,255], [51,51,255],
                         [127, 0,255],[255, 255, 255]]
            # red, orange, yellow,green,light_blue, deep blue, purple, white,
            color = color_map[skill_id]

        else:
            color = [255,0,0]

        fill_coords(img, tri_fn, [int(color_i*agent_confidence) for color_i in color])

    img = downsample(img, subdivs)

    return img


def plot_grid(height, width, grid_list, move_map, skill_map=None, filter_rare=True, skill_confidence=False):
    """

    :param height:
    :param width:
    :param grid_list:
    :param move_map:
    :param skill_map:
    :param filter_rare:
    :param skill_confidence: if False: use action to compute confidence; if True, use skill selection to compute
    :return:
    """

    width_px = width * TILE_PIXELS
    height_px = height * TILE_PIXELS

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    for j in range(0, height):
        for i in range(0, width):
            cell = grid_list[j*width+i]

            # agent_here = np.array_equal(agent_pos, (i, j))
            tile_img = render_tile(
                cell,
                tile_size=TILE_PIXELS,
                move=move_map[j,i],
                skill=skill_map[j,i],
                filter_rare = filter_rare,
                skill_confidence = skill_confidence,
            )

            ymin = j * TILE_PIXELS
            ymax = (j + 1) * TILE_PIXELS
            xmin = i * TILE_PIXELS
            xmax = (i + 1) * TILE_PIXELS
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img


