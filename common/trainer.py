from typing import Dict
import logging
import random

import torch
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

class MyTrainer(Trainer):
    def __init__(
        self,
        model= None,
        args= None,
        data_collator= None,
        train_dataset= None,
        eval_dataset= None,
        tokenizer= None,
        model_init= None,
        compute_metrics = None,
        callbacks= None,
        optimizers= (None, None),
        preprocess_logits_for_metrics = None,
        metric_class=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.metric_class = metric_class
        
    
    # TODO optimize for multiprocess
    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys= None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        with torch.no_grad():
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
                )
            
            self.model.eval()

            f1s = []
            ious = []

            for start_idx in range(0, len(self.eval_dataset), self._train_batch_size):
                end_idx = min(len(self.eval_dataset), start_idx + self._train_batch_size)
                
                batch = {
                    'pixel_values': torch.tensor(self.eval_dataset['pixel_values'][start_idx:end_idx]).to(self.model.device),
                    'input_ids': torch.tensor(self.eval_dataset['input_ids'][start_idx:end_idx]).to(self.model.device),
                    'attention_mask': torch.tensor(self.eval_dataset['attention_mask'][start_idx:end_idx]).to(self.model.device),
                    'qformer_input_ids': torch.tensor(self.eval_dataset['qformer_input_ids'][start_idx:end_idx]).to(self.model.device),
                    'qformer_attention_mask': torch.tensor(self.eval_dataset['qformer_attention_mask'][start_idx:end_idx]).to(self.model.device),
                }
                
                outputs = self.model.generate(
                        **batch,
                        do_sample = False,
                        num_beams=5,
                        max_length=128,
                        min_length=1,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                    )
                
                gen_metrics = self.metric_class.compute_metrics(
                    EvalPrediction(predictions=outputs.cpu(), label_ids=self.eval_dataset['label_ids'][start_idx:end_idx]),
                    tokenized_pred = True,
                    verbose = True,
                )
                f1s.append(gen_metrics['f1_micro'])
                ious.append(gen_metrics['iou_micro'])

                if random.random() < 0.3:
                    rand_idx = random.randint(0, len(outputs)-1)
                    gen_text = self.metric_class.tokenizer.decode(outputs[rand_idx].cpu())
                    logging.info(f'- Generation example: ,{gen_text}')

                del batch, outputs
        
        torch.cuda.empty_cache()
        
        gen_metrics = {
            'gen_f1': sum(f1s) / len(f1s),
            'gen_iou': sum(ious) / len(ious)
        }
        metrics.update(gen_metrics)
        
        logging.info('==================================')
        logging.info(f'* Evaluation result: {metrics}')
        
        return metrics