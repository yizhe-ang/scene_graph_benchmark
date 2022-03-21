N_GPUS=1

# srun -p dsta --mpi=pmi2 --gres=gpu:${N_GPUS} -n1 --ntasks-per-node=${N_GPUS} --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
# python \
# -m torch.distributed.launch \
# --nproc_per_node=${N_GPUS} \
# --master_port=1234 \
# -m tools.train_sg_net \
# --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml


srun -p dsta --mpi=pmi2 --gres=gpu:${N_GPUS} -n1 --ntasks-per-node=${N_GPUS} --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
python -m tools.train_sg_net \
--config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_grcnn.yaml
