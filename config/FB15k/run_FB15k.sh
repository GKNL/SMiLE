#!/bin/bash


REPO_DIR='/SMiLE'

dataset='FB15k'
gpu_id=0

data_path="${REPO_DIR}/data"
#pretrained_embeddings=$data_path/../act_$dataset\_mult_500.out
#pretrained_method='compgcn'
pretrained_embeddings="${data_path}/${dataset}/${dataset}.emd"
outdir="${REPO_DIR}/output/${dataset}-1"
mkdir $outdir

# Pretrain Setting
n_heads=4
n_layers=4
n_pretrain_epochs=15
pretrain_batch_size=1024
pretrain_checkpoint=4

# Schema Setting
# relation_weight=1
schema_weight=1
topNfilters=-700
use_schema=True

# Finetune Setting
ft_n_epochs=15
ft_batch_size=256
ft_checkpoint=500
walk_type='bfs'
num_walks_per_node=1
beam_width=6
max_length=6
gcn_option=no_gcn
node_edge_composition_func=mult
ft_input_option='last4_cat'
path_option='shortest'
is_pre_trained=True


python main.py \
    --gpu_id $gpu_id \
    --data_name $dataset \
    --data_path $data_path \
    --outdir $outdir \
    --pretrained_embeddings $pretrained_embeddings \
    --n_epochs $n_pretrain_epochs \
    --batch_size $pretrain_batch_size \
    --checkpoint $pretrain_checkpoint \
    --schema_weight $schema_weight\
    --n_layers $n_layers \
    --n_heads $n_heads \
    --gcn_option $gcn_option \
    --node_edge_composition_func $node_edge_composition_func \
    --ft_input_option $ft_input_option \
    --path_option $path_option \
    --ft_n_epochs $ft_n_epochs \
    --ft_batch_size $ft_batch_size \
    --ft_checkpoint $ft_checkpoint \
    --num_walks_per_node $num_walks_per_node \
    --beam_width $beam_width \
    --max_length $max_length \
    --walk_type $walk_type \
    --topNfilters $topNfilters \
    --is_pre_trained \
    --use_schema
    #>> $log_file