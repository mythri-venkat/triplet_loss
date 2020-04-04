from argparse import ArgumentParser

import os

import h5py
import json
import numpy as np
import tensorflow as tf

import common
from models import Trinet

parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', default="../marketroot/",
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--dataset', default="data/market1501_query.csv",
    help='Path to the dataset csv file to be embedded.')

# Optional

parser.add_argument(
    '--image_root', default="../Market-1501-v15.09.15/",type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel data loading.')

parser.add_argument(
    '--batch_size', default=32, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')

parser.add_argument(
    '--filename', default=None,
    help='Name of the HDF5 file in which to store the embeddings, relative to'
         ' the `experiment_root` location. If omitted, appends `_embeddings.h5`'
         ' to the dataset name.')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')



parser.add_argument(
    '--net_input_height', default=256, type=common.positive_int,
    help='Height of the input directly fed into the network.')

parser.add_argument(
    '--net_input_width', default=128, type=common.positive_int,
    help='Width of the input directly fed into the network.')



def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args([])

    # Possibly auto-generate the output filename.
    if args.filename is None:
        basename = os.path.basename(args.dataset)
        args.filename = os.path.splitext(basename)[0] + '_embeddings.h5'
    args.filename = os.path.join(args.experiment_root, args.filename)

   
    _, data_fids = common.load_dataset(args.dataset, args.image_root)

    net_input_size = (args.net_input_height, args.net_input_width)
    # pre_crop_size = (args.pre_crop_height, args.pre_crop_width)

    # Setup a tf Dataset containing all images.
    dataset = tf.data.Dataset.from_tensor_slices(data_fids)

    # Convert filenames to actual image tensors.
    dataset = dataset.map(
        lambda fid: common.fid_to_image(
            fid, tf.constant('dummy'), image_root=args.image_root,
            image_size=net_input_size),
        num_parallel_calls=args.loading_threads)

    
    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming.
    dataset = dataset.prefetch(1)

    

    model = Trinet(args.embedding_dim)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,args.train_iterations - args.decay_start_iteration, 0.001)
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.experiment_root, max_to_keep=10)

    
    ckpt.restore(manager.latest_checkpoint)

    with h5py.File(args.filename, 'w') as f_out:
        emb_storage = np.zeros((len(data_fids) , args.embedding_dim), np.float32)
        start_idx = 0 
        for images,fids,pids in dataset:
            emb = model(images,training=False)
            emb_storage[start_idx:start_idx+len(emb)]=emb
            start_idx+=args.batch_size
        emb_dataset = f_out.create_dataset('emb', data=emb_storage)
if __name__ == '__main__':
    main()
