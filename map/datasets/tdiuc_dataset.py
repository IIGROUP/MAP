import torch

import json
import csv
import os

from PIL import Image
from ..transforms import keys_to_transforms

class TDIUCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        transform_keys: list,
        image_size: int,
        max_text_len=40,
        tokenizer=None,
    ):
        super().__init__()

        assert split in ["train", "val"]
        self.split = split

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer

        ans2id = {}
        with open('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/sample_answerkey.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                ans2id[row[0]] = int(row[1])
        print('the answer list size:', min(ans2id.values()), max(ans2id.values()))

        if split == 'train':
            ques = json.load(open('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/Questions/OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
            ans = json.load(open('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/Annotations/mscoco_train2014_annotations.json', 'r'))['annotations']
        else:
            ques = json.load(open('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/Questions/OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
            ans = json.load(open('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/Annotations/mscoco_val2014_annotations.json', 'r'))['annotations']
        quesid2ansid = {}
        for anno in ans:
            quesid = anno['question_id']
            imgid = anno['image_id']
            answer = anno['answers'][0]['answer']
            # assert answer in ans2id, answer
            if answer in ans2id: # 有问题的QA和正常的QA: 325/538543
                answerid = ans2id[answer]
                quesid2ansid[quesid] = {'ansid':answerid, 'imgid':imgid}

        self.qa_list = []
        for question in ques:
            quesid = question['question_id']
            imgid = question['image_id']
            sentence = question['question']
            if quesid in quesid2ansid:
                ansid = quesid2ansid[quesid]['ansid']
                assert imgid == quesid2ansid[quesid]['imgid']
                imgpath = self.imgid2path(imgid)
                self.qa_list.append({'quesid':quesid, 'question':sentence, 'imgpath':imgpath, 'ansid':ansid})

    def __len__(self):
        return len(self.qa_list)

    def imgid2path(self, imgid):
        if self.split == 'train':
            return os.path.join('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/Images/train2014', 'COCO_train2014_'+str(imgid).rjust(12,'0')+'.jpg')
        else:
            return os.path.join('/cognitive_comp/wangjunjie/data/downstream_Data/TDIUC/Images/val2014', 'COCO_val2014_'+str(imgid).rjust(12,'0')+'.jpg')
        
    def get_image(self, imgpath):
        image = Image.open(imgpath).convert("RGBA")
        img_tensor = [tr(image) for tr in self.transforms]
        return img_tensor

    def get_text(self, ques):
        encoding = self.tokenizer(
            ques,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return encoding

    def __getitem__(self, index):
        item = self.qa_list[index]
        img = self.get_image(item['imgpath'])
        ques = self.get_text(item['question'])
        text_ids = torch.tensor(ques['input_ids'])
        text_labels = torch.zeros_like(text_ids)
        text_masks = torch.tensor(ques['attention_mask'])
        return {
            'qid':item['quesid'], 
            'image':img, 
            'text_ids':text_ids,
            'text_labels':text_labels,
            'text_masks':text_masks,
            'ans_label':torch.tensor(item['ansid']),
        }
