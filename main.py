import math

import os, networkx as nx
import time

import numpy as np
from PIL import Image, ImageDraw

DIR_PATH = os.path.dirname(__file__)
IMAGE_PATH = DIR_PATH + "/resources/"


def init_graph(image_path, marked_image_path, color_space, neighbour_noise_factor, inverted_lambda):
    image = Image.open(image_path, "r").convert(color_space)
    width, height = image.size
    image_marked = Image.open(marked_image_path, "r").convert(color_space)
    G = nx.Graph()
    object_pixels = []
    background_pixels = []
    max_weight = 2**20
    base = 0

    gradient_intensity = Image.new("HSV", (width, height))
    print("adding nodes and n-link weights")
    for i in range(height):
        offset = 0
        for j in range(width):
            image_pixel = image.getpixel((j, i))
            image_marked_pixel = image_marked.getpixel((j, i))
            G.add_node(base + offset, x=j, y=i, feature=image_pixel)
            if color_space == "RGB":
                if abs(image_pixel[0] - image_marked_pixel[0]) >= 20 and image_marked_pixel[0] >= 240:
                    object_pixels.append(G.nodes[base + offset])
                    # print(f"object image_pixel: {image_pixel}, marked: {image_marked_pixel}")
                if abs(image_pixel[2] - image_marked_pixel[2]) >= 20 and image_marked_pixel[2] >= 240:
                    background_pixels.append(G.nodes[base + offset])
                    # print(f"background image_pixel: {image_pixel}, marked: {image_marked_pixel}")
            elif color_space == "HSV":
                if euclidian_distance(image_pixel, image_marked_pixel) >= 20 and image_marked_pixel[1] >= 200 and \
                        image_marked_pixel[2] >= 200:
                    if image_marked_pixel[0] <= 20 or image_marked_pixel[0] >= 230:
                        # print(f"object image_pixel: {image_pixel}, marked: {image_marked_pixel}")
                        object_pixels.append(G.nodes[base + offset])
                    elif 100 <= image_marked_pixel[0] <= 200:
                        # print(f"background image_pixel: {image_pixel}, marked: {image_marked_pixel}")
                        background_pixels.append(G.nodes[base + offset])
            if i > 0:
                weight = inverted_lambda * gen_weight(G.nodes[base + offset], G.nodes[base + offset - width], neighbour_noise_factor)
                gradient_intensity.putpixel((j, i), (0, int(255 * weight / inverted_lambda), 255))
                """
                if weight > max_weight:
                    max_weight = weight
                """
                G.add_edge(base + offset, base + offset - width, weight=weight)
            if j > 0:
                weight = inverted_lambda * gen_weight(G.nodes[base + offset], G.nodes[base + offset - 1], neighbour_noise_factor)
                gradient_intensity.putpixel((j, i), (0, 255 - int(255 * weight / inverted_lambda), 255))
                """
                if weight > max_weight:
                    max_weight = weight
                """
                G.add_edge(base + offset, base + offset - 1, weight=weight)
            offset += 1
        base += width

    gradient_intensity.show()
    gradient_intensity.convert('RGB').save(IMAGE_PATH+"koala/boundary_"+str(neighbour_noise_factor)+"_lambda"+str(inverted_lambda)+"_"+str(time.time())+".png")
    print("calculating gaussian distribution of object and background pixels")
    o_segments = threshold_seeds(object_pixels, 'feature')
    b_segments = threshold_seeds(background_pixels, 'feature')

    # """
    seeds_segments_image = Image.new("HSV", (width, height))
    pixelMap_seeds = seeds_segments_image.load()
    for keys in o_segments.keys():
        # print(f"object, key={keys}, mean={o_segments[keys]['mean']}, cov={o_segments[keys]['cov']}")
        for pixel in o_segments[keys]["values"]:
            pixelMap_seeds[pixel['x'], pixel['y']] = tuple(o_segments[keys]["mean"].astype(int))
    for keys in b_segments.keys():
        # print(f"background, key={keys}, mean={b_segments[keys]['mean']}, cov={b_segments[keys]['cov']}")
        for pixel in b_segments[keys]["values"]:
            pixelMap_seeds[pixel['x'], pixel['y']] = tuple(b_segments[keys]["mean"].astype(int))
    seeds_segments_image.show()
    seeds_segments_image.convert('RGB').save(IMAGE_PATH+"koala/seeds_"+str(time.time())+".png")
    # """

    regional_cor_image = Image.new("RGB", (width, height))
    pixelMap = regional_cor_image.load()

    print("calculating t-link weights")
    index_source = base
    G.add_nodes_from([index_source, index_source + 1])  # add S and T
    base = 0
    for i in range(height):
        offset = 0
        for j in range(width):
            ob_prob = regional_prob_object_segments(o_segments, G.nodes[base + offset], 'feature')
            bg_prob = regional_prob_object_segments(b_segments, G.nodes[base + offset], 'feature')

            bg_weight = max_weight if ob_prob == 0 else -math.log(ob_prob)
            ob_weight = max_weight if bg_prob == 0 else -math.log(bg_prob)
            # print(f"x:{j}, y:{i}, ob_prob: {ob_prob}, ob_weight:{ob_weight}, bg_prop:{bg_prob}, bg_weight:{
            # bg_weight}")

            G.add_edge(base + offset, index_source, weight=ob_weight)
            G.add_edge(base + offset, index_source + 1, weight=bg_weight)

            pixelMap[j, i] = (int(ob_prob/(ob_prob+bg_prob) * 255), 0, int(bg_prob/(ob_prob+bg_prob) * 255))

            offset += 1
        base += width

    regional_cor_image.show()
    regional_cor_image.convert('RGB').save(IMAGE_PATH+"koala/regional_2sqrt"+str(time.time())+".png")

    print("adding t-links for seeds")
    for node in object_pixels:
        index_node = node['y'] * width + node['x']
        G.edges[index_node, index_source]["weight"] = 6 * max_weight
        G.edges[index_node, index_source + 1]["weight"] = 0
    for node in background_pixels:
        index_node = node['y'] * width + node['x']
        G.edges[index_node, index_source]["weight"] = 0
        G.edges[index_node, index_source + 1]["weight"] = 6 * max_weight

    # print(G.edges.data(data=True))
    G.to_undirected()

    out = image.copy()
    return G, index_source, index_source + 1, out


def gen_weight(pixel1, pixel2, neighbour_noise_factor):
    if pixel1['x'] == pixel2['x'] and pixel1['y'] == pixel2['y']:
        return 0
    # spatial_factor = 1 / (euclidian_distance((pixel1['x'], pixel1['y']), (pixel2['x'], pixel2['y'])))
    spatial_factor = 1  # since we only consider 4-neighbourhood
    range_factor = math.exp(-(euclidian_distance(pixel1['feature'], pixel2['feature']) ** 2) / neighbour_noise_factor)
    return spatial_factor * range_factor


def log_table(length):
    return (-math.log((i + 1) / length) for i in range(length))


def euclidian_distance(tup1, tup2):
    assert len(tup1) == len(tup2), "Euclidian distance only measurable for vectors of equating dimension"
    d = 0
    for i in range(len(tup1)):
        d += (tup2[i] - tup1[i]) ** 2
    return math.sqrt(d)


def regional_prob_object_alt(o, b, p):
    if len(o) < 1 or len(b) < 1:
        return 0.5
    mean_distance_o = (sum_euclidian_distance(o, p, 'feature') / len(o))
    mean_distance_b = (sum_euclidian_distance(b, p, 'feature') / len(b))
    return mean_distance_b / (mean_distance_o + mean_distance_b)


def regional_prob_object(o_mean, o_cov, b_mean, b_cov, p):
    o = euclidian_distance(o_mean, p['feature']) / o_cov
    b = euclidian_distance(b_mean, p['feature']) / b_cov
    return b / (o + b)


def regional_prob_object_segments(segments, pixel, attr):
    return sum(segments[key]['prop'] * math.exp(-(euclidian_distance(tuple(segments[key]['mean']), pixel[attr]) /
                                                  (2*math.sqrt(segments[key]['cov'])))) for key in segments.keys())


def sum_euclidian_distance(tupel_set, tup, attr):
    return sum(euclidian_distance(t[attr], tup[attr]) for t in tupel_set)


def multivariate_gaussian_distribution(tuples, attr):
    if not tuples:
        return None, None

    dim = len(tuples[0][attr])
    mean = np.zeros(dim)
    tuple_count = len(tuples)
    for i in range(dim):
        mean[i] = sum(tup[attr][i] for tup in tuples) / tuple_count
    cov = sum(euclidian_distance(mean, tup[attr]) ** 2 for tup in tuples)
    return mean, math.sqrt(cov)


def univariate_gaussian_distribution(tuples, attr, index):
    if not tuples:
        return 0, 0
    tuple_count = len(tuples)
    if index is not None and index > -1:
        mean = sum(tup[attr][index] for tup in tuples) / tuple_count
        cov = sum((mean - tup[attr][index]) ** 2 for tup in tuples)
        return mean, math.sqrt(cov)


def threshold_seeds(tuples, attr):
    if not tuples:
        return []

    segments = {}
    size = len(tuples)

    # first separate black values
    near_black = list(tup for tup in tuples if tup[attr][2] < 10)
    rest = list(tup for tup in tuples if tup not in near_black)
    if len(near_black) > 0:
        mean_black, cov_black = multivariate_gaussian_distribution(near_black, attr)
        segments[0] = {"mean": mean_black, "cov": max(cov_black, 0.00000001), "values": near_black, "prop": len(near_black) / size}

    # edge case: red segment
    # near_red = list(tup for tup in rest if tup[attr][0] < 25 or tup[attr][0] > 230)
    # rest = list(tup for tup in rest if tup not in near_red)
    # if len(near_red) > 0:
    #     mean_red, cov_red = multivariate_gaussian_distribution(near_red, attr)
    #     segments[1] = {"mean": mean_red, "cov": max(cov_red, 0.00000001), "values": near_red, "prop": len(near_red) / size}

    threshold_seeds_rec(segments, rest, attr, 0, 255, size)

    return segments


def threshold_seeds_rec(segments, tuples, attr, l, r, size):
    if not tuples:
        return None
    # print(f"threshold_seeds_rec, l={l}, r={r}")

    if r - l <= 5:
        mean, cov = multivariate_gaussian_distribution(tuples, attr)
        if mean is not None and cov is not None:
            segments[r] = {"mean": mean, "cov": max(cov, 0.00000001), "values": tuples, "prop": len(tuples) / size}
        return segments
    min_index, left, right, optimal_threshold = opt_threshold(tuples, attr, l, r)
    if min_index < 5:
        thres_l, thres_r = optimal_threshold - (r - l) // 10, optimal_threshold + (r - l) // 10
        while thres_r - thres_l > 1:
            min_index, left, right, optimal_threshold = opt_threshold(tuples, attr, thres_l, thres_r)
            thres_l, thres_r = optimal_threshold - (thres_r - thres_l) // 10, optimal_threshold + (
                    thres_r - thres_l) // 10

        threshold_seeds_rec(segments, left, attr, l, optimal_threshold, size)
        threshold_seeds_rec(segments, right, attr, optimal_threshold + 1, r, size)  # optimal_threshold + 1 < r
    else:
        mean, cov = multivariate_gaussian_distribution(left, attr)
        if mean is not None and cov is not None:
            segments[r] = {"mean": mean, "cov": max(cov, 0.00000001), "values": left, "prop": len(left) / size}
    return segments


def opt_threshold(tuples, attr, l, r):
    cov_sum = float('inf')
    index = 6
    ret_threshold = -1
    left, right = [], []
    if l >= r:
        return index, left, right, ret_threshold
    for i in range(1, 6):
        threshold = l + i * (r - l) // 5
        left_temp = list(tup for tup in tuples if tup[attr][0] <= threshold)
        right_temp = list(tup for tup in tuples if tup[attr][0] > threshold)
        mean_l, cov_l = univariate_gaussian_distribution(left_temp, attr, index=0)
        mean_r, cov_r = univariate_gaussian_distribution(right_temp, attr, index=0)
        # print(f"threshold={threshold}, mean_l={mean_l}, cov_l={cov_l}, mean_r={mean_r}, cov_r={cov_r}")

        if cov_l + cov_r <= cov_sum:
            cov_sum = cov_l + cov_r
            index = i
            left = left_temp
            right = right_temp
            ret_threshold = threshold
    return index, left, right, ret_threshold


if __name__ == '__main__':
    space = "HSV"
    image_filename = 'koala.jpg'
    image_name = image_filename.split('.')[0]
    image_marked_filename = 'koala_marked2.jpg'
    neighbour_noise = 2500
    inverted_lambda = 10
    g, s, t, out = init_graph(IMAGE_PATH + image_filename, IMAGE_PATH + image_marked_filename, color_space=space,
                              neighbour_noise_factor=neighbour_noise, inverted_lambda=inverted_lambda)
    print("determining minimum cut")
    cut_value, partition = nx.minimum_cut(g, s, t, capacity='weight')
    reachable, non_reachable = partition

    pixelMap = out.load()
    width, height = out.size

    for node_index in non_reachable:
        if node_index != t:
            pixelMap[node_index % width, node_index / width] = (0, 0, 0)

    out.show()
    out.convert('RGB').save(IMAGE_PATH+"koala/result_noise"+str(neighbour_noise)+"_lambda"+str(inverted_lambda)+"_"+str(time.time())+".png")

    # out.save(IMAGE_PATH + image_name + "_segmented")
