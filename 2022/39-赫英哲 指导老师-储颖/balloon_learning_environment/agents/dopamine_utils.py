# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for Dopamine-based agents."""

import os.path as osp
import pathlib
import pickle
from typing import Any, Dict, Callable

from absl import logging
import tensorflow as tf


def _make_checkpoint_filename(checkpoint_dir: str,
                              iteration_number: int) -> str:
  return osp.join(checkpoint_dir, f'checkpoint_{iteration_number:05d}.pkl')


def _is_checkpoint_name(file_name: str) -> bool:
  path = pathlib.Path(file_name)
  if path.suffix != '.pkl':
    return False

  parts = path.stem.split('_')
  if len(parts) != 2:
    return False

  return parts[0] == 'checkpoint' and len(parts[1]) == 5


def _get_checkpoint_iteration(checkpoint_name: str) -> int:
  path = pathlib.Path(checkpoint_name)
  parts = path.stem.split('_')
  return int(parts[1])


def save_checkpoint(checkpoint_dir: str,
                    iteration_number: int,
                    bundle_fn: Callable[[str, int], Any]) -> None:
  """Save a checkpoint using the provided bundling function."""
  # Try to create checkpoint directory if it doesn't exist.
  try:
    tf.io.gfile.makedirs(checkpoint_dir)
  except tf.errors.PermissionDeniedError:
    # If it already exists, ignore exception.
    pass

  bundle = bundle_fn(checkpoint_dir, iteration_number)
  if bundle is None:
    logging.warning('Unable to checkpoint to %s at iteration %d.',
                    checkpoint_dir, iteration_number)
    return

  filename = _make_checkpoint_filename(checkpoint_dir, iteration_number)
  with tf.io.gfile.GFile(filename, 'w') as fout:
    pickle.dump(bundle, fout)


def load_checkpoint(
    checkpoint_dir: str,
    iteration_number: int,
    unbundle_fn: Callable[[str, int, Dict[Any, Any]], bool]) -> None:
  """Load a checkpoint using the provided unbundling function."""
  filename = _make_checkpoint_filename(checkpoint_dir, iteration_number)
  if not tf.io.gfile.exists(filename):
    logging.warning('Unable to restore bundle from %s', filename)
    return

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    bundle = pickle.load(fin)
  if not unbundle_fn(checkpoint_dir, iteration_number, bundle):
    logging.warning('Call to parent `unbundle` failed.')


def get_latest_checkpoint(checkpoint_dir: str) -> int:
  """Find the episode ID of the latest checkpoint, if any."""
  glob = osp.join(checkpoint_dir, 'checkpoint_*.pkl')
  def extract_episode(x):
    return int(x[x.rfind('checkpoint_') + 11:-4])

  try:
    checkpoint_files = tf.io.gfile.glob(glob)
  except tf.errors.NotFoundError:
    logging.warning('Unable to reload checkpoint at %s', checkpoint_dir)
    return -1

  try:
    latest_episode = max(extract_episode(x) for x in checkpoint_files)
  except ValueError:
    return -1
  return latest_episode


def clean_up_old_checkpoints(checkpoint_dir: str,
                             episode_number: int,
                             checkpoint_duration: int = 50) -> None:
  """Removes the most recent stale checkpoint.

  Args:
    checkpoint_dir: Directory where checkpoints are stored.
    episode_number: Current episode number.
    checkpoint_duration: How long (in terms of episodes) a checkpoint should
      last.
  """
  for file_name in tf.io.gfile.listdir(checkpoint_dir):
    if _is_checkpoint_name(file_name):
      iteration = _get_checkpoint_iteration(file_name)
      if iteration < episode_number - checkpoint_duration:
        tf.io.gfile.remove(osp.join(checkpoint_dir, file_name))
