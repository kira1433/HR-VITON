python train_generator.py -b 4 -j 8 --cuda True --name test --gpu_ids 0 --fp16 --tocg_checkpoint ./checkpoints/tocg_final.pth --gen_checkpoint ./eval_models/weights/gen.pth --dis_checkpoint ./eval_models/weights/discriminator_mtviton.pth --occlusion