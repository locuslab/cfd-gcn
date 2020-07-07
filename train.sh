# Prediction experiments

export BATCH_SIZE=16; CUDA_VISIBLE_DEVICES=0 mpirun -np $((BATCH_SIZE+1)) --oversubscribe python train.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --su2-config coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-5 --data-dir data/NACA0012_interpolate --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 > stdout.log

export BATCH_SIZE=16; CUDA_VISIBLE_DEVICES=0 mpirun -np $((BATCH_SIZE+1)) --oversubscribe python train.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-5 --data-dir data/NACA0012_interpolate/



# Generalization experiments

export BATCH_SIZE=16; CUDA_VISIBLE_DEVICES=0 mpirun -np $((BATCH_SIZE+1)) --oversubscribe python train.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --su2-config coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-5 --data-dir data/NACA0012_machsplit_noshock --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 > stdout.log

export BATCH_SIZE=16; CUDA_VISIBLE_DEVICES=0 mpirun -np $((BATCH_SIZE+1)) --oversubscribe python train.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-5 --data-dir data/NACA0012_machsplit_noshock/
