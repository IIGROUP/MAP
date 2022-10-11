# pretrain
# _config = {
#     'exp_name': "mlm_itm_con",
#     'seed': 0,
#     'datasets': ["coco", "vg"],
#     'loss_names': {
#             "itm": 1,
#             "mlm": 1,
#             "con": 1,
#             "vqa": 0,
#             "nlvr2": 0,
#             "snli": 0,
#             "tdiuc": 0,
#             "irtr": 0,
#         },
#     'batch_size': 4096,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

#     # Image setting
#     'image_size': 288,
#     'draw_false_image': 0, # hitm
#     'image_only': False,
#     'vit': 'ViT-B/16',
#     'patch_size': 16,
#     'train_transform_keys': ["clip"],
#     'val_transform_keys': ["clip"],
#     'input_image_embed_size': 768,

#     # Text Setting
#     'tokenizer': "roberta-base",
#     'vocab_size': 50265,
#     'input_text_embed_size': 768,
#     'max_text_len': 50,
#     'vqav2_label_size': 3129,
#     'tdiuc_label_size': 1480,
#     'mlm_prob': 0.15,
#     'draw_false_text': 0,
#     'whole_word_masking': True,

#     # Transformer Setting 
#     'num_layers': 6,
#     'mlp_ratio': 4,
#     'drop_rate': 0.1,
#     'num_top_layer': 6,
#     'hidden_size': 768,
#     'num_heads': 12,

#     # Optimizer Setting
#     'optim_type': "adamw",
#     'weight_decay': 0.01,
#     'decay_power': 1,
#     'end_lr': 0,
#     'learning_rate': 1e-5,
#     'val_check_interval': 1.0,
#     'lr_mult_head': 5,
#     'lr_mult_cross_modal': 5,
#     'lr_mult_gaussian': 5,
#     'max_epoch': 10,
#     'max_steps': 20000,
#     'warmup_steps': 0.1,

#     # PL Trainer Setting
#     'resume_from': None, # load interrupted ckpt
#     'fast_dev_run': False, # for debug
#     'test_only': False,

#     # below params varies with the environment
#     'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
#     'log_dir': "result",
#     'per_gpu_batchsize': 14,  # you should define this manually with per_gpu_batch_size=#
#     'num_gpus': 8,
#     'num_nodes': 4,
#     'load_path': "",
#     'num_workers': 8,
#     'precision': 32,

#     # for retrieval
#     'get_recall_metric': False,
#     'candidate_N': 128,

#     # gaussian
#     'gaussian': True,
#     'sample_num': 5,
#     'mu_num': 1, # sample_num+mu_num
#     'margin_loss': True,
#     'margin_value': 300,
#     'margin_weight': 0.01,
    
#     # contrast
#     'negative_scale': 1/200,
#     'shift': 4,

# }

# VQA2.0
# _config = {
#     'exp_name': "finetune_vqa",
#     'seed': 0,
#     'datasets': ["vqa"],
#     'loss_names': {
#             "itm": 0,
#             "mlm": 0,
#             "con": 0,
#             "vqa": 1,
#             "nlvr2": 0,
#             "snli": 0,
#             "tdiuc": 0,
#             "irtr": 0,
#         },
#     'batch_size': 512,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

#     # Image setting
#     'image_size': 384,
#     'draw_false_image': 0,
#     'image_only': False,
#     'vit': 'ViT-B/16',
#     'patch_size': 16,
#     'train_transform_keys': ["clip_randaug"],
#     'val_transform_keys': ["clip_test"],
#     'input_image_embed_size': 768,

#     # Text Setting
#     'tokenizer': "roberta-base",
#     'vocab_size': 50265,
#     'input_text_embed_size': 768,
#     'max_text_len': 50,
#     'vqav2_label_size': 3129,
#     'tdiuc_label_size': 1480,
#     'mlm_prob': 0.15,
#     'draw_false_text': 0,
#     'whole_word_masking': True,

#     # Transformer Setting 
#     'num_layers': 6,
#     'mlp_ratio': 4,
#     'drop_rate': 0.1,
#     'num_top_layer': 6,
#     'hidden_size': 768,
#     'num_heads': 12,

#     # Optimizer Setting
#     'optim_type': "adamw",
#     'weight_decay': 0.01,
#     'decay_power': 1,
#     'end_lr': 0,
#     'learning_rate': 5e-6,
#     'val_check_interval': 0.1,
#     'lr_mult_head': 50,
#     'lr_mult_cross_modal': 5,
#     'lr_mult_gaussian': 40,
#     'max_epoch': 10,
#     'max_steps': None,
#     'warmup_steps': 0.1,

#     # PL Trainer Setting
#     'resume_from': None, # load interrupted ckpt
#     'fast_dev_run': False, # for debug
#     'test_only': False,

#     # below params varies with the environment
#     'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
#     'log_dir': "result",
#     'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
#     'num_gpus': 8,
#     'num_nodes': 1,
#     'load_path': "", # pretrain weigth path
#     'num_workers': 8,
#     'precision': 32,

#     # for retrieval
#     'get_recall_metric': False,
#     'candidate_N': 128,

#     # gaussian
#     'gaussian': True,
#     'sample_num': 5,
#     'mu_num': 1, # sample_num+mu_num
#     'margin_loss': True,
#     'margin_value': 300,
#     'margin_weight': 0.01,
    
#     # contrast
#     'negative_scale': 1/200,
#     'shift': 4,

# }

# NLVR2
# _config = {
#     'exp_name': "finetune_nlvr2",
#     'seed': 0,
#     'datasets': ["nlvr2"],
#     'loss_names': {
#             "itm": 0,
#             "mlm": 0,
#             "con": 0,
#             "vqa": 0,
#             "nlvr2": 1,
#             "snli": 0,
#             "tdiuc": 0,
#             "irtr": 0,
#         },
#     'batch_size': 256,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

#     # Image setting
#     'image_size': 384,
#     'draw_false_image': 0,
#     'image_only': False,
#     'vit': 'ViT-B/16',
#     'patch_size': 16,
#     'train_transform_keys': ["clip_randaug"],
#     'val_transform_keys': ["clip_test"],
#     'input_image_embed_size': 768,

#     # Text Setting
#     'tokenizer': "roberta-base",
#     'vocab_size': 50265,
#     'input_text_embed_size': 768,
#     'max_text_len': 50,
#     'vqav2_label_size': 3129,
#     'tdiuc_label_size': 1480,
#     'mlm_prob': 0.15,
#     'draw_false_text': 0,
#     'whole_word_masking': True,

#     # Transformer Setting 
#     'num_layers': 6,
#     'mlp_ratio': 4,
#     'drop_rate': 0.1,
#     'num_top_layer': 6,
#     'hidden_size': 768,
#     'num_heads': 12,

#     # Optimizer Setting
#     'optim_type': "adamw",
#     'weight_decay': 0.01,
#     'decay_power': 1,
#     'end_lr': 0,
#     'learning_rate': 1e-5,
#     'val_check_interval': 1.0,
#     'lr_mult_head': 10,
#     'lr_mult_cross_modal': 5,
#     'lr_mult_gaussian': 20,
#     'max_epoch': 10,
#     'max_steps': None,
#     'warmup_steps': 0.1,

#     # PL Trainer Setting
#     'resume_from': None, # load interrupted ckpt
#     'fast_dev_run': False, # for debug
#     'test_only': False,

#     # below params varies with the environment
#     'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
#     'log_dir': "result",
#     'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
#     'num_gpus': 8,
#     'num_nodes': 1,
#     'load_path': "",
#     'num_workers': 8,
#     'precision': 32,

#     # for retrieval
#     'get_recall_metric': False,
#     'candidate_N': 128,

#     # gaussian
#     'gaussian': True,
#     'sample_num': 5,
#     'mu_num': 1, # sample_num+mu_num
#     'margin_loss': True,
#     'margin_value': 300,
#     'margin_weight': 0.01,
    
#     # contrast
#     'negative_scale': 1/200,
#     'shift': 4,

# }

# SNLI
# _config = {
#     'exp_name': "finetune_snli",
#     'seed': 0,
#     'datasets': ["snli"],
#     'loss_names': {
#             "itm": 0,
#             "mlm": 0,
#             "con": 0,
#             "vqa": 0,
#             "nlvr2": 0,
#             "snli": 1,
#             "tdiuc": 0,
#             "irtr": 0,
#         },
#     'batch_size': 64,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

#     # Image setting
#     'image_size': 384,
#     'draw_false_image': 0,
#     'image_only': False,
#     'vit': 'ViT-B/16',
#     'patch_size': 16,
#     'train_transform_keys': ["clip_randaug"],
#     'val_transform_keys': ["clip_test"],
#     'input_image_embed_size': 768,

#     # Text Setting
#     'tokenizer': "roberta-base",
#     'vocab_size': 50265,
#     'input_text_embed_size': 768,
#     'max_text_len': 50,
#     'vqav2_label_size': 3129,
#     'tdiuc_label_size': 1480,
#     'mlm_prob': 0.15,
#     'draw_false_text': 0,
#     'whole_word_masking': True,

#     # Transformer Setting 
#     'num_layers': 6,
#     'mlp_ratio': 4,
#     'drop_rate': 0.1,
#     'num_top_layer': 6,
#     'hidden_size': 768,
#     'num_heads': 12,

#     # Optimizer Setting
#     'optim_type': "adamw",
#     'weight_decay': 0.01,
#     'decay_power': 1,
#     'end_lr': 0,
#     'learning_rate': 2e-6,
#     'val_check_interval': 1.0,
#     'lr_mult_head': 10,
#     'lr_mult_cross_modal': 5,
#     'lr_mult_gaussian': 40,
#     'max_epoch': 5,
#     'max_steps': None,
#     'warmup_steps': 0.1,

#     # PL Trainer Setting
#     'resume_from': None, # load interrupted ckpt
#     'fast_dev_run': False, # for debug
#     'test_only': False,

#     # below params varies with the environment
#     'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
#     'log_dir': "result",
#     'per_gpu_batchsize': 8,  # you should define this manually with per_gpu_batch_size=#
#     'num_gpus': 8,
#     'num_nodes': 1,
#     'load_path': "",
#     'num_workers': 8,
#     'precision': 32,

#     # for retrieval
#     'get_recall_metric': False,
#     'candidate_N': 128,

#     # gaussian
#     'gaussian': True,
#     'sample_num': 5,
#     'mu_num': 1, # sample_num+mu_num
#     'margin_loss': True,
#     'margin_value': 300,
#     'margin_weight': 0.01,
    
#     # contrast
#     'negative_scale': 1/200,
#     'shift': 4,

# }


# f30k
# _config = {
#     'exp_name': "finetune_irtr_f30k",
#     'seed': 0,
#     'datasets': ["f30k"],
#     'loss_names': {
#             "itm": 0.5,
#             "mlm": 0,
#             "con": 0.5,
#             "vqa": 0,
#             "nlvr2": 0,
#             "snli": 0,
#             "tdiuc": 0,
#             "irtr": 1,
#         },
#     'batch_size': 512,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

#     # Image setting
#     'image_size': 384,
#     'draw_false_image': 0,
#     'image_only': False,
#     'vit': 'ViT-B/16',
#     'patch_size': 16,
#     'train_transform_keys': ["clip_randaug"],
#     'val_transform_keys': ["clip_test"],
#     'input_image_embed_size': 768,

#     # Text Setting
#     'tokenizer': "roberta-base",
#     'vocab_size': 50265,
#     'input_text_embed_size': 768,
#     'max_text_len': 50,
#     'vqav2_label_size': 3129,
#     'tdiuc_label_size': 1480,
#     'mlm_prob': 0.15,
#     'draw_false_text': 15,
#     'whole_word_masking': True,

#     # Transformer Setting 
#     'num_layers': 6,
#     'mlp_ratio': 4,
#     'drop_rate': 0.1,
#     'num_top_layer': 6,
#     'hidden_size': 768,
#     'num_heads': 12,

#     # Optimizer Setting
#     'optim_type': "adamw",
#     'weight_decay': 0.01,
#     'decay_power': 1,
#     'end_lr': 0,
#     'learning_rate': 5e-6,
#     'val_check_interval': 1.0,
#     'lr_mult_head': 5,
#     'lr_mult_cross_modal': 5,
#     'lr_mult_gaussian': 10,
#     'max_epoch': 10,
#     'max_steps': None,
#     'warmup_steps': 0.1,

#     # PL Trainer Setting
#     'resume_from': None, # load interrupted ckpt
#     'fast_dev_run': False, # for debug
#     'test_only': True,

#     # below params varies with the environment
#     'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
#     'log_dir': "result",
#     'per_gpu_batchsize': 1,  # you should define this manually with per_gpu_batch_size=#
#     'num_gpus': 8,
#     'num_nodes': 1,
#     'load_path': "",
#     'num_workers': 8,
#     'precision': 32,

#     # for retrieval
#     'get_recall_metric': True,
#     'candidate_N': 128,

#     # gaussian
#     'gaussian': True,
#     'sample_num': 5,
#     'mu_num': 1, # sample_num+mu_num
#     'margin_loss': True,
#     'margin_value': 300,
#     'margin_weight': 0.01,
    
#     # contrast
#     'negative_scale': 1/200,
#     'shift': 4,

# }


# coco
_config = {
    'exp_name': "finetune_irtr_coco",
    'seed': 0,
    'datasets': ["coco"],
    'loss_names': {
            "itm": 0.5,
            "mlm": 0,
            "con": 0.5,
            "vqa": 0,
            "nlvr2": 0,
            "snli": 0,
            "tdiuc": 0,
            "irtr": 1,
        },
    'batch_size': 512,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 384,
    'draw_false_image': 0,
    'image_only': False,
    'vit': 'ViT-B/16',
    'patch_size': 16,
    'train_transform_keys': ["clip_randaug"],
    'val_transform_keys': ["clip_test"],
    'input_image_embed_size': 768,

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'input_text_embed_size': 768,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'tdiuc_label_size': 1480,
    'mlm_prob': 0.15,
    'draw_false_text': 15,
    'whole_word_masking': True,

    # Transformer Setting 
    'num_layers': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'num_top_layer': 6,
    'hidden_size': 768,
    'num_heads': 12,

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 5e-6,
    'val_check_interval': 1.0,
    'lr_mult_head': 5,
    'lr_mult_cross_modal': 5,
    'lr_mult_gaussian': 10,
    'max_epoch': 10,
    'max_steps': None,
    'warmup_steps': 0.1,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': True,

    # below params varies with the environment
    'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 1,  # you should define this manually with per_gpu_batch_size=#
    'num_gpus': 8,
    'num_nodes': 1,
    'load_path': "",
    'num_workers': 8,
    'precision': 32,

    # for retrieval
    'get_recall_metric': True,
    'candidate_N': 256,

    # gaussian
    'gaussian': True,
    'sample_num': 5,
    'mu_num': 1, # sample_num+mu_num
    'margin_loss': True,
    'margin_value': 300,
    'margin_weight': 0.01,
    
    # contrast
    'negative_scale': 1/200,
    'shift': 4,

}
