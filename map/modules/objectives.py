import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import numpy as np
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather, AllGather_multi


# def margin_entropy_loss(margin, logsigma): # 见PUM和ReID那两篇文章
#     feat_dim = logsigma.shape[-1]
#     entropy = torch.mean(float(feat_dim / 2 * (np.log(2 * np.pi) + 1)) + torch.sum(logsigma, -1) / 2)
#     zero = torch.zeros_like(entropy)
#     loss = torch.max(margin - entropy, zero)
#     return loss

def margin_entropy_loss(margin, logsigma): # 见PUM和ReID那两篇文章
    feat_dim = logsigma.shape[-1]
    entropy = float(feat_dim / 2 * (np.log(2 * np.pi) + 1)) + torch.sum(logsigma, -1) / 2
    zero = torch.zeros_like(entropy)
    loss = torch.max(margin - entropy, zero)
    loss = torch.mean(loss)
    return loss

def Wasserstein2(mu1, sigma1, mu2, sigma2): # 2W距离，传入图片和文本的均值和标准差
    bs1 = mu1.shape[0]
    bs2 = mu2.shape[0]
    mu1 = torch.stack([mu1]*bs2, dim=1)
    sigma1 = torch.stack([sigma1]*bs2, dim=1)
    mu2 = torch.stack([mu2]*bs1, dim=0)
    sigma2 = torch.stack([sigma2]*bs1, dim=0)
    p1 = torch.sum(torch.pow(mu1 - mu2, 2), dim=-1)
    p2 = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=-1)
    return p1+p2, p1

def compute_contrast(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False, contrast=True)
    
    image_mu = infer['image_mu'][:, 0]
    image_sigma = torch.exp(infer['image_logsigma'][:, 0])
    text_mu = infer['text_mu'][:, 0]
    text_sigma = torch.exp(infer['text_logsigma'][:, 0])

    pl_module.log('con/img_sigma_mean', torch.mean(image_sigma), on_step=True)
    pl_module.log('con/txt_sigma_mean', torch.mean(text_sigma), on_step=True)
    
    bs = image_mu.shape[0]
    phase = "train" if pl_module.training else "val"

    # gather
    # allgather = AllGather_multi.apply
    # image_mu = allgather(image_mu)
    # text_mu = allgather(text_mu)
    # image_sigma = allgather(image_sigma)
    # text_sigma = allgather(text_sigma)

    W2_distance, mu_distance = Wasserstein2(image_mu, image_sigma, text_mu, text_sigma)
    similarity = (-pl_module.negative_scale * W2_distance + pl_module.shift) / pl_module.temp

    labels = torch.arange(bs).to(similarity.device)
    loss = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    
    pl_module.log(f"contrast/{phase}/loss", loss)
    pl_module.log("temperature", pl_module.temp)

    ret = {'contrast_loss': loss}

    return ret

def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    if pl_module.gaussian:
        mlm_logits = mlm_logits.reshape((pl_module.sample_num+1), -1, mlm_logits.shape[-2], mlm_logits.shape[-1])
        mlm_loss = []
        for i in range(pl_module.sample_num+1):
            mlm_loss.append(F.cross_entropy(mlm_logits[i].view(-1, pl_module.hparams.config["vocab_size"]), mlm_labels.view(-1), ignore_index=-100))
        mlm_loss = sum(mlm_loss) / (pl_module.sample_num+1)
        mlm_logits = torch.mean(mlm_logits, dim=0)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'])
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
            mlm_labels.view(-1),
            ignore_index=-100,
        )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    if pl_module.gaussian and pl_module.margin:
        ret['mlm_loss'] = ret['mlm_loss'] + pl_module.margin_weight * margin_loss

    return ret

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])

    if pl_module.gaussian:
        itm_logits = itm_logits.reshape((pl_module.sample_num+1), -1, itm_logits.shape[-1])
        itm_loss = []
        for i in range(pl_module.sample_num+1):
            itm_loss.append(F.cross_entropy(itm_logits[i], itm_labels.long()))   
        itm_loss = sum(itm_loss) / (pl_module.sample_num+1)
        itm_logits = torch.mean(itm_logits, dim=0)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'][:, 0]) 
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'][:, 0])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:    
        itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    if pl_module.gaussian and pl_module.margin:
        ret['itm_loss'] = ret['itm_loss'] + pl_module.margin_weight * margin_loss

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    if pl_module.gaussian:
        len_targets = int(len(vqa_logits) / (pl_module.sample_num+pl_module.mu_num))
    else:
        len_targets = len(vqa_logits)
    vqa_targets = torch.zeros(
        len_targets, pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    if pl_module.gaussian:
        vqa_logits = vqa_logits.reshape((pl_module.sample_num+pl_module.mu_num), -1, vqa_logits.shape[-1])
        vqa_loss = []
        for i in range(pl_module.sample_num+pl_module.mu_num):
            vqa_loss.append(F.binary_cross_entropy_with_logits(vqa_logits[i], vqa_targets) * vqa_targets.shape[1])
        vqa_loss = sum(vqa_loss) / (pl_module.sample_num+pl_module.mu_num)
        vqa_logits = torch.mean(vqa_logits, dim=0)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'][:, 0]) # 对整体调整还是每个元素的标准差都要比较？
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'][:, 0])
            # margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'])
            # margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:
        vqa_loss = (
            F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
            * vqa_targets.shape[1]
        )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    # if len(pl_module.vqa_logits_list) < 20:
    #     pl_module.vqa_logits_list.append(vqa_logits)
    #     print(len(pl_module.vqa_logits_list))
    # if len(pl_module.vqa_logits_list) == 20:
    #     res = torch.cat(pl_module.vqa_logits_list).cpu().numpy()
    #     np.save(os.path.join(pl_module.log_dir, 'vqa_logits.npy'), res)

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    if pl_module.gaussian and pl_module.margin:
        ret['vqa_loss'] = ret['vqa_loss'] + pl_module.margin_weight * margin_loss

    return ret

def compute_tdiuc(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    tdiuc_logits = pl_module.tdiuc_classifier(infer["cls_feats"])

    tdiuc_labels = batch["ans_label"]

    if pl_module.gaussian:
        tdiuc_logits = tdiuc_logits.reshape((pl_module.sample_num+pl_module.mu_num), -1, tdiuc_logits.shape[-1])
        tdiuc_loss = []
        for i in range(pl_module.sample_num+pl_module.mu_num):
            tdiuc_loss.append(F.cross_entropy(tdiuc_logits[i], tdiuc_labels.view(-1)))
        tdiuc_loss = sum(tdiuc_loss) / (pl_module.sample_num+pl_module.mu_num)
        tdiuc_logits = torch.mean(tdiuc_logits, dim=0)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'][:, 0]) # 对整体调整还是每个元素的标准差都要比较？
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'][:, 0])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:
        tdiuc_loss = F.cross_entropy(tdiuc_logits, tdiuc_labels.view(-1))

    ret = {
        "tdiuc_loss": tdiuc_loss,
        "tdiuc_logits": tdiuc_logits,
        "tdiuc_labels": tdiuc_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_tdiuc_loss")(ret["tdiuc_loss"])
    accuracy = getattr(pl_module, f"{phase}_tdiuc_accuracy")(
        ret["tdiuc_logits"], ret["tdiuc_labels"]
    )
    pl_module.log(f"tdiuc/{phase}/loss", loss)
    pl_module.log(f"tdiuc/{phase}/accuracy", accuracy)

    if pl_module.gaussian and pl_module.margin:
        ret['tdiuc_loss'] = ret['tdiuc_loss'] + pl_module.margin_weight * margin_loss

    return ret


def compute_snli(pl_module, batch): # 视觉推理，判断文本和图片关系，三分类
    infer = pl_module.infer(
        batch, mask_text=False, mask_image=False, 
    )
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()

    if pl_module.gaussian:
        snli_logits = snli_logits.reshape((pl_module.sample_num+1), -1, snli_logits.shape[-1])
        snli_loss = []
        for i in range(pl_module.sample_num+1):
            snli_loss.append(F.cross_entropy(snli_logits[i], snli_labels.view(-1)))     
        snli_loss = sum(snli_loss) / (pl_module.sample_num+1)
        snli_logits = torch.mean(snli_logits, dim=0)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'][:, 0]) 
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'][:, 0])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:
        snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{phase}_snli_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"snli/{phase}/loss", loss)
        pl_module.log(f"snli/{phase}/accuracy", acc)

        if pl_module.gaussian and pl_module.margin:
            ret['snli_loss'] = ret['snli_loss'] + pl_module.margin_weight * margin_loss
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_accuracy")(
                ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            )
            pl_module.log(f"snli/dev/loss", dev_loss)
            pl_module.log(f"snli/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"snli/test/loss", test_loss)
            pl_module.log(f"snli/test/accuracy", test_acc)

    return ret


def compute_nlvr2(pl_module, batch): # 两张图片和一个文本，判断是否正确，二分类
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    
    if pl_module.gaussian:
        nlvr2_logits = nlvr2_logits.reshape((pl_module.sample_num+1), -1, nlvr2_logits.shape[-1])
        nlvr2_loss = []
        for i in range(pl_module.sample_num+1):
            nlvr2_loss.append(F.cross_entropy(nlvr2_logits[i], nlvr2_labels.view(-1)))     
        nlvr2_loss = sum(nlvr2_loss) / (pl_module.sample_num+1)
        nlvr2_logits = torch.mean(nlvr2_logits, dim=0)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer1['image_logsigma'][:, 0]) 
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer1['text_logsigma'][:, 0])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)

        if pl_module.gaussian and pl_module.margin:
            ret['nlvr2_loss'] = ret['nlvr2_loss'] + pl_module.margin_weight * margin_loss
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch): # 检索
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"] # 15个负样本
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        },
        irtr = True,
    )
    answer = torch.zeros(_bs).to(text_ids).long() # 对每组样本中第0个是正样本对

    # contrast
    con_img_mu = infer['con_img_mu'][:,0].reshape(_bs, false_len+1, -1)
    con_img_sigma = torch.exp(infer['con_img_logsigma'])[:,0].reshape(_bs, false_len+1, -1)
    con_txt_mu = infer['con_txt_mu'][:,0].reshape(_bs, false_len+1, -1)
    con_txt_sigma = torch.exp(infer['con_txt_logsigma'])[:,0].reshape(_bs, false_len+1, -1)
    W2_distance = []
    for i in range(_bs):
        W2_distance.append(Wasserstein2(con_img_mu[i,0:1], con_img_sigma[i,0:1], con_txt_mu[i], con_txt_sigma[i])[0].squeeze())
    W2_distance = torch.stack(W2_distance, dim=0)
    similarity = -pl_module.negative_scale * W2_distance + pl_module.shift
    con_loss = F.cross_entropy(similarity, answer)

    # itm head
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    if pl_module.gaussian:
        score = score.reshape((pl_module.sample_num+1), _bs, false_len+1)
        irtr_loss = []
        for i in range(pl_module.sample_num+1):
            irtr_loss.append(F.cross_entropy(score[i], answer))   
        irtr_loss = sum(irtr_loss) / (pl_module.sample_num+1)
        if pl_module.margin:
            margin_loss1 = margin_entropy_loss(pl_module.margin_value, infer['image_logsigma'][:, 0]) 
            margin_loss2 = margin_entropy_loss(pl_module.margin_value, infer['text_logsigma'][:, 0])
            margin_loss = (margin_loss1 + margin_loss2) / 2
            pl_module.log(f"margin_loss", margin_loss, on_step=True)
    else:    
        score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
        irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "con_loss": con_loss,
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
    pl_module.log(f"irtr/{phase}/con_loss", con_loss)

    return ret

@torch.no_grad()
def img_feat_retrieval(pl_module, images):
    device = images.device
    image_embeds = pl_module.vit_model(images)
    image_embeds = pl_module.cross_modal_image_transform(image_embeds)
    image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
    extend_image_masks = pl_module.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)
    img_mu, img_logsigma, _ = pl_module.con_img_gau_encoder(image_embeds, mask=extend_image_masks)
    img_dist = torch.stack([img_mu[:,0], torch.exp(img_logsigma[:,0])], dim=1)
    return img_dist

@torch.no_grad()
def text_feat_retrieval(pl_module, text_batch):
    text_ids = text_batch["text_ids"]
    text_masks = text_batch["text_masks"]

    text_embeds = pl_module.text_transformer.embeddings(input_ids=text_ids)
    device = text_embeds.device
    input_shape = text_masks.size()
    extend_text_masks = pl_module.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
    for layer in pl_module.text_transformer.encoder.layer:
        text_embeds = layer(text_embeds, extend_text_masks)[0]
    text_embeds = pl_module.cross_modal_text_transform(text_embeds)
    txt_mu, txt_logsigma, _ = pl_module.con_txt_gau_encoder(text_embeds, mask=extend_text_masks)
    txt_dist = torch.stack([txt_mu[:,0], torch.exp(txt_logsigma[:,0])], dim=1)
    return txt_dist
    

# @torch.no_grad()
# def compute_irtr_recall(pl_module): # irtr的预测部分
#     text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
#     text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
#     text_loader = torch.utils.data.DataLoader(
#         text_dset,
#         batch_size=32,
#         num_workers=pl_module.hparams.config["num_workers"],
#         pin_memory=True,
#         collate_fn=functools.partial(
#             text_dset.collate,
#             mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
#         ),
#     )

#     image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
#         image_only=True
#     )
#     image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
#     image_loader = torch.utils.data.DataLoader(
#         image_dset,
#         batch_size=1,
#         num_workers=pl_module.hparams.config["num_workers"],
#         pin_memory=True,
#         collate_fn=functools.partial(
#             image_dset.collate,
#             mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
#         ),
#     )

#     text_preload = list()
#     for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
#         text_preload.append(
#             {
#                 "text_ids": _b["text_ids"].to(pl_module.device),
#                 "text_masks": _b["text_masks"].to(pl_module.device),
#                 "text_labels": _b["text_labels"].to(pl_module.device),
#                 "img_index": _b["img_index"],
#             }
#         )

#     tiids = list()
#     for pre in text_preload:
#         tiids += pre["img_index"]
#     tiids = torch.tensor(tiids)
#     text_ids = torch.cat([txt['text_ids'] for txt in text_preload], dim=0)
#     text_masks = torch.cat([txt['text_masks'] for txt in text_preload], dim=0)
#     text_labels = torch.cat([txt['text_labels'] for txt in text_preload], dim=0)

#     image_preload = list()
#     for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
#         image_preload.append((_b['image'][0], _b["img_index"][0]))
#     img_feat = torch.cat([im[0] for im in image_preload], dim=0).to(device=text_ids.device)

#     img_dist = []
#     txt_dist = []
#     for img in tqdm.tqdm(image_preload, desc="img "):
#         img_dist.append(img_feat_retrieval(pl_module, img[0].to(device=text_ids.device)))
#     img_dist = torch.cat(img_dist, dim=0)
#     for txt in tqdm.tqdm(text_preload, desc='txt '):
#         txt_dist.append(text_feat_retrieval(pl_module, {"text_ids": txt["text_ids"], "text_masks": txt["text_masks"], "text_labels": txt["text_labels"]}))
#     txt_dist = torch.cat(txt_dist, dim=0)

#     # W2_distance, _ = Wasserstein2(img_dist[:,0], img_dist[:,1], txt_dist[:,0], txt_dist[:,1]) # OOM
#     W2_distance = []
#     for i in range(len(img_dist)):
#         W2_distance.append(Wasserstein2(img_dist[i:i+1,0], img_dist[i:i+1,1], txt_dist[:,0], txt_dist[:,1])[0])
#     W2_distance = torch.cat(W2_distance, dim=0)
#     print('img/txt num:', W2_distance.shape)
#     N = pl_module.candidate_N
#     i2t_topk = (-W2_distance).topk(N, dim=1).indices #(1000, N)
#     t2i_topk = (-W2_distance).topk(N, dim=0).indices #(N, 5000)
#     similarity = -pl_module.negative_scale * W2_distance + pl_module.shift

#     scores = torch.full_like(W2_distance, -1000).float()
#     scores0 = torch.full_like(W2_distance, -1000).float()
#     iids = []    

#     for i,img in enumerate(tqdm.tqdm(image_preload, desc='img ')):
#         _im, _iid = img
#         ids = i2t_topk[i]    
#         img = _im.repeat(N, 1, 1, 1).to(device=text_ids.device)
#         batch = {
#             "text_ids": text_ids[ids],
#             "text_masks": text_masks[ids],
#             "text_labels": text_labels[ids],
#         }
#         infer = pl_module.infer(batch, mask_text=False, mask_image=False, img=img)
#         logits = pl_module.rank_output(infer["cls_feats"])
#         if pl_module.gaussian:
#             logits = logits.reshape((pl_module.sample_num+1), -1, logits.shape[-1])
#             logits = torch.mean(logits, dim=0).squeeze()
#         sim = similarity[i,ids]
#         scores[i,ids] = logits
#         scores0[i,ids] = sim
#         iids.append(_iid)

#     iids = torch.tensor(iids)
#     iids = iids.view(-1)

#     topk10 = scores.topk(10, dim=1)
#     topk5 = scores.topk(5, dim=1)
#     topk1 = scores.topk(1, dim=1)
#     topk10_iids = tiids[topk10.indices]
#     topk5_iids = tiids[topk5.indices]
#     topk1_iids = tiids[topk1.indices]
    
#     tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
#     tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
#     tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
#     print('tr_logits:', tr_r1, tr_r5, tr_r10)

#     topk100 = scores0.topk(N, dim=1)
#     topk10 = scores0.topk(10, dim=1)
#     topk5 = scores0.topk(5, dim=1)
#     topk1 = scores0.topk(1, dim=1)
#     topk100_iids = tiids[topk100.indices]
#     topk10_iids = tiids[topk10.indices]
#     topk5_iids = tiids[topk5.indices]
#     topk1_iids = tiids[topk1.indices]
    
#     tr_r100_ = (iids.unsqueeze(1) == topk100_iids).float().max(dim=1)[0].mean()
#     tr_r10_ = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
#     tr_r5_ = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
#     tr_r1_ = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
#     print('tr_sim:', tr_r1_, tr_r5_, tr_r10_, tr_r100_)

#     for i in tqdm.tqdm(range(len(text_ids)), desc='text '):
#         ids = t2i_topk[:,i]
#         batch = {
#             "text_ids": text_ids[i:i+1].repeat(N, 1),
#             "text_masks": text_masks[i:i+1].repeat(N, 1),
#             "text_labels": text_labels[i:i+1].repeat(N, 1),
#         }
#         infer = pl_module.infer(batch, mask_text=False, mask_image=False, img=img_feat[ids])
#         logits = pl_module.rank_output(infer["cls_feats"])
#         if pl_module.gaussian:
#             logits = logits.reshape((pl_module.sample_num+1), -1, logits.shape[-1])
#             logits = torch.mean(logits, dim=0).squeeze()
#         sim = similarity[ids,i]
#         scores[ids,i] = logits
#         scores0[ids,i] = sim

#     topk10 = scores.topk(10, dim=0)
#     topk5 = scores.topk(5, dim=0)
#     topk1 = scores.topk(1, dim=0)
#     topk10_iids = iids[topk10.indices]
#     topk5_iids = iids[topk5.indices]
#     topk1_iids = iids[topk1.indices]

#     ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
#     ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
#     ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
#     print('ir_logits:', ir_r1, ir_r5, ir_r10)

#     topk100 = scores0.topk(N, dim=0)
#     topk10 = scores0.topk(10, dim=0)
#     topk5 = scores0.topk(5, dim=0)
#     topk1 = scores0.topk(1, dim=0)
#     topk100_iids = iids[topk100.indices]
#     topk10_iids = iids[topk10.indices]
#     topk5_iids = iids[topk5.indices]
#     topk1_iids = iids[topk1.indices]

#     ir_r100_ = (tiids.unsqueeze(0) == topk100_iids).float().max(dim=0)[0].mean()
#     ir_r10_ = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
#     ir_r5_ = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
#     ir_r1_ = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
#     print('ir_sim:', ir_r1_, ir_r5_, ir_r10_, ir_r100_)

#     return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)

@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=16,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    #TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )
            if pl_module.gaussian:
                score = score.reshape((pl_module.sample_num+1), -1, score.shape[-1])
                score = torch.mean(score, dim=0)[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name, log_dir):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        with open(os.path.join(log_dir, "vqa_submit_%s.json"%model_name), "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
