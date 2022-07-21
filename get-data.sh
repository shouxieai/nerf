#!/bin/bash

mkdir ckpt data imgs rotate360 videos
wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz --output-document=data/tiny_nerf_data.npz