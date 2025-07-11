# coding=utf-8
# Copyright 2021 The Rliable Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rliable import library as rly
from rliable import metrics

import collections
import numpy as np
import warnings
import logging
import seaborn as sns
from matplotlib import rcParams
from matplotlib import rc

warnings.filterwarnings('default')
RAND_STATE = np.random.RandomState(42)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sns.set_style("white")
# Matplotlib params
rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rc('text', usetex=False)

StratifiedBootstrap = rly.StratifiedBootstrap
IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)


def normalize_score(score, min_score, max_score=1):
  norm_score = (score - min_score)/(max_score - min_score)
  return norm_score


def CI(bootstrap_dist, stat_val=None, alpha=0.05, is_pivotal=False):
    """
    Get the bootstrap confidence interval for a given distribution.
    Args:
      bootstrap_distribution: numpy array of bootstrap results.
      stat_val: The overall statistic that this method is attempting to
        calculate error bars for. Default is None.
      alpha: The alpha value for the confidence intervals.
      is_pivotal: if true, use the pivotal (reverse percentile) method.
        If false, use the percentile method.
    Returns:
      (low, high): The lower and upper limit for `alpha` x 100% CIs.
      val: The median value of the bootstrap distribution if `stat_val` is None
        else `stat_val`.
    """
    # Adapted from https://pypi.org/project/bootstrapped
    if is_pivotal:
      assert stat_val is not None, 'Please pass the statistic for a pivotal'
      'confidence interval'
      low = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
      val = stat_val
      high = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (alpha / 2.))
    else:
      low = np.percentile(bootstrap_dist, 100 * (alpha / 2.))
      val = np.percentile(bootstrap_dist, 50)
      high = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
    return (low, high), val

RANDOM_SCORE = {'random2-4': 60773.60171508789, 'random4-6':  46458.919845581055, 'random6-8': 36298.69444274902, 'random8-10': 38716.394790649414}

HUMAN_SCORE = 1.0

def create_score_dict_atari_100k(main_df, normalization=True,
                                 evaluation_key='eval_average_return'):
    """Creates a dictionary of scores."""
    score_dict = {}
    for key, df in main_df.items():
        score_dict[key] = df[evaluation_key].values
    if normalization:
        score_dict = normalize_score(score_dict, RANDOM_SCORE, HUMAN_SCORE)
    return score_dict


def convert_to_matrix(x):
  return np.stack([x[k] for k in sorted(x.keys())], axis=-1)

def get_scores(df, normalization=True, eval='Final'):
    score_dict_df = create_score_dict_atari_100k(df, normalization=normalization)
    score_matrix = convert_to_matrix(score_dict_df)
    median, mean = MEDIAN(score_matrix), MEAN(score_matrix)
    print('{}: Median: {}, Mean: {}'.format(eval, median, mean))
    return score_dict_df, score_matrix

def subsample_scores_mat(score_mat, num_samples=5, replace=False):
  total_samples, num_games = score_mat.shape
  subsampled_scores = np.empty((num_samples, num_games))
  for i in range(num_games):
    indices = np.random.choice(total_samples, size=num_samples, replace=replace)
    subsampled_scores[:, i] = score_mat[indices, i]
  return subsampled_scores

def get_rank_matrix(score_dict, n=10000, algorithms=None):
  arr = []
  if algorithms is None:
    algorithms = sorted(score_dict.keys())
  for alg in algorithms:
    arr.append(subsample_scores_mat(
        score_dict[alg], num_samples=n, replace=True))
  X = np.stack(arr, axis=0)
  num_algs, _, num_tasks = X.shape
  all_mat = []
  for task in range(num_tasks):
    # Sort based on negative scores as rank 0 corresponds to minimum value,
    # rank 1 corresponds to second minimum value when using lexsort.
    task_x = -X[:, :, task]
    # This is done to randomly break ties.
    rand_x = np.random.random(size=task_x.shape)
    # Last key is the primary key,
    indices = np.lexsort((rand_x, task_x), axis=0)
    mat = np.zeros((num_algs, num_algs))
    for rank in range(num_algs):
      cnts = collections.Counter(indices[rank])
      mat[:, rank] = np.array([cnts[i]/n for i in range(num_algs)])
    all_mat.append(mat)
  all_mat = np.stack(all_mat, axis=0)
  return all_mat
