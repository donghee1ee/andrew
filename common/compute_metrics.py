import logging
import pickle
import random
import numpy as np

import torch
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.utils import to_one_hot
from utils.eval import calculate_iou_batch, calculate_iou

def regression_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    bs = labels.shape[0]

    # TODO - parallel
    ious = []
    for i in range(bs):
        iou = calculate_iou(labels[i], preds[i])
        ious.append(iou)
    
    mean_iou = np.mean(np.array(ious))

    for _ in range(3):
        i = random.randint(0, bs-1)
        logging.info('========================')
        logging.info(f'** GT: {labels[i]}')
        logging.info(f'** Prediction: {preds[i]}')
        iou = calculate_iou(labels[i], preds[i])
        logging.info(f'** IoU: {iou}')

    return {'mean_iou': mean_iou}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    if len(preds.shape) == 2: # 2D
        preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy() >= 0.5
    else: # 3D
        preds = torch.sigmoid(torch.tensor(pred.predictions[0])).numpy() >= 0.5

    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    iou_macro = jaccard_score(labels, preds, average='macro') # TODO macro iou?
    iou_micro = jaccard_score(labels, preds, average='micro')

    result = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': acc,
        'iou_macro': iou_macro,
        'iou_micro': iou_micro,
    }

    logging.info(f'* Evaluation result: {result}')

    return result

class DecoderEvalMetrics():
    
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.ingr2id = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb')).word2idx
        label_ids2ingr_class = dict()
        for entry in eval_dataset:
            label_ids2ingr_class[tuple(entry['label_ids'])] = entry['ingredient_int']
        self.label_ids2ingr_class = label_ids2ingr_class

    def map_to_classes(self, batch_tokens, max_len=20):
        ingredient_text = self.tokenizer.batch_decode(batch_tokens)
        
        # Process all ingredients in a batch together
        batch_ingr_ids = []
        for ingrs in ingredient_text:
            ingr_text = [ingr.strip().replace(' ', '_') for ingr in ingrs.split(',')]
            ingr_ids = [self.ingr2id.get(ingr, None) for ingr in ingr_text if ingr in self.ingr2id]
            # batch_ingr_ids.append(ingr_ids)

            # Pad the list to ensure consistent length
            if max_len > len(ingr_ids):
                padded_ingr_ids = ingr_ids + [self.ingr2id.get("<pad>", -1)] * (max_len - len(ingr_ids))
            else:
                padded_ingr_ids = ingr_ids
            batch_ingr_ids.append(padded_ingr_ids[:max_len])  # Ensures the list is not longer than max_len

        return batch_ingr_ids

    def compute_metrics(self, pred, tokenized_pred=False, verbose=True):
        labels = pred.label_ids # text_output ids
        target_ingr = []
        for label in labels:
            ingrs = self.label_ids2ingr_class[tuple(label)]
            target_ingr.append(ingrs)
        
        target_ingr = torch.tensor(target_ingr) # one-hot already
        target_ingr = to_one_hot(target_ingr)

        if tokenized_pred:
            pred_ingr = self.map_to_classes(pred.predictions)
        else:
            pred_ingr = self.map_to_classes(pred.predictions[0].argmax(-1))
        pred_ingr = to_one_hot(torch.tensor(pred_ingr))
        
        f1_micro = f1_score(target_ingr, pred_ingr, average='micro')
        f1_macro = f1_score(target_ingr, pred_ingr, average='macro')
        iou_micro = jaccard_score(target_ingr, pred_ingr, average='micro')
        iou_macro = jaccard_score(target_ingr, pred_ingr, average='macro')
        # acc = accuracy_score(target_ingr, pred_ingr)

        result = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            # 'accuracy': acc,
            'iou_macro': iou_macro,
            'iou_micro': iou_micro,
        }

        if verbose:
            logging.info(f'* Evaluation result: {result}')

        return result