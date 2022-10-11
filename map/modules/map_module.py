import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import os

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import swin_transformer as swin
from . import heads, objectives, map_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel
from .attention_block import Block
from .PDE import DisTrans

class MAPTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.is_clip= (not 'swin' in config['vit'])
        
        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights) # 设置一些初始化参数，比如均值、标准差
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained('/apdcephfs/share_1367250/auroraji/pretrained_weight/roberta_base')
                else:
                    BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model('/apdcephfs/share_1367250/auroraji/pretrained_weight/clip-vit/ViT-B-16.pt', resolution_after=resolution_after)
        else:
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained('/apdcephfs/share_1367250/auroraji/pretrained_weight/roberta_base')
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.gaussian = config['gaussian']
        if self.gaussian:

            self.img_gau_encoder = DisTrans(768, 12)
            self.txt_gau_encoder = DisTrans(768, 12)
            self.img_gau_encoder.apply(objectives.init_weights)
            self.txt_gau_encoder.apply(objectives.init_weights)

            self.sample_num = config['sample_num']
            self.mu_num = config['mu_num']
            self.margin = config['margin_loss']
            self.margin_value = config['margin_value']
            self.margin_weight = config['margin_weight']

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"]) # 一层mlp
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["con"] > 0:
            self.negative_scale = config["negative_scale"]
            self.shift = config["shift"]
            self.temp = nn.Parameter(torch.ones([]) * 0.07)

            self.con_img_gau_encoder = DisTrans(768, 12)
            self.con_txt_gau_encoder = DisTrans(768, 12)
            self.con_img_gau_encoder.apply(objectives.init_weights)
            self.con_txt_gau_encoder.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, before=288, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["tdiuc"] > 0:
            vs = self.hparams.config["tdiuc_label_size"]
            self.tdiuc_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.tdiuc_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :] # itm_score.fc是hs×2，取第一维作为检索的分数  
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            self.candidate_N = config['candidate_N']
            for p in self.itm_score.parameters():
                p.requires_grad = False

        map_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, before=resolution_after, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)

        self.log_dir = config['log_dir']

        # p-value
        # self.vqa_logits_list = []
        # self.snli_logits_list = []
        # self.nlvr2_logits_list = []

    def gaussian_modeling(
        self,
        image_embeds,
        extend_image_masks,
        text_embeds,
        extend_text_masks,
    ):
        img_mu, img_logsigma, _ = self.img_gau_encoder(image_embeds, mask=extend_image_masks)
        if self.training:
            self.log('img_sigma_mean', torch.mean(torch.exp(img_logsigma)), on_step=True)
        z = [img_mu] * self.mu_num
        for i in range(self.sample_num):
            eps = torch.randn(img_mu.shape[0], img_mu.shape[1], img_mu.shape[2], device=img_mu.device)
            z1 = img_mu + torch.exp(img_logsigma) * eps
            z.append(z1)
        image_embeds = torch.cat(z)

        txt_mu, txt_logsigma, _ = self.txt_gau_encoder(text_embeds, mask=extend_text_masks)
        if self.training:
            self.log('txt_sigma_mean', torch.mean(torch.exp(txt_logsigma)), on_step=True)
        z = [txt_mu] * self.mu_num
        for i in range(self.sample_num):
            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2], device=txt_mu.device)
            z1 = txt_mu + torch.exp(txt_logsigma) * eps
            z.append(z1)
        text_embeds = torch.cat(z)

        return image_embeds, text_embeds, img_mu, img_logsigma, txt_mu, txt_logsigma

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        contrast=False,
        irtr=False,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device) # [bs,len] -> [bs,1,1,len]
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        image_embeds = self.vit_model(img)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        if contrast:
            img_mu, img_logsigma, _ = self.con_img_gau_encoder(image_embeds, mask=extend_image_masks)
            txt_mu, txt_logsigma, _ = self.con_txt_gau_encoder(text_embeds, mask=extend_text_masks)
            ret = {
                "image_mu": img_mu, 
                "text_mu": txt_mu,
                "image_logsigma": img_logsigma,
                "text_logsigma": txt_logsigma,
            }
            return ret

        if irtr:
            con_img_mu, con_img_logsigma, _ = self.con_img_gau_encoder(image_embeds, mask=extend_image_masks)
            con_txt_mu, con_txt_logsigma, _ = self.con_txt_gau_encoder(text_embeds, mask=extend_text_masks)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        img_logsigma = None
        txt_logsigma = None
        img_mu = None
        txt_mu = None
        if self.gaussian:
            y, x, img_mu, img_logsigma, txt_mu, txt_logsigma = self.gaussian_modeling(y, extend_image_masks, x, extend_text_masks)

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        if irtr:
            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "image_logsigma": img_logsigma,
                "text_logsigma": txt_logsigma,
                "con_img_mu": con_img_mu,
                "con_img_logsigma": con_img_logsigma,
                "con_txt_mu": con_txt_mu,
                "con_txt_logsigma": con_txt_logsigma,
            }
        else:
            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "image_logsigma": img_logsigma,
                "text_logsigma": txt_logsigma,
            }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Contrast Learning
        if "con" in self.current_tasks:
            ret.update(objectives.compute_contrast(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # TDIUC VQA
        if "tdiuc" in self.current_tasks:
            ret.update(objectives.compute_tdiuc(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx): # 得到每一步的损失
        map_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs): # 一代结束后计算指标
        map_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        map_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        map_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        map_utils.set_task(self)
        if "irtr" in self.current_tasks:
            return
            
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.log_dir)
        map_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return map_utils.set_schedule(self)
