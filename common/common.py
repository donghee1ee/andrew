import json
import logging

from transformers import TrainerCallback

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

class PrinterCallback(TrainerCallback): 
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:  # Log at intervals defined by logging_steps
            # Access model and dataloader
            model = kwargs['model']
            dataloader = kwargs['dataloader']

            # Assuming you have a function to get predictions and labels
            predictions, labels = self.get_predictions_and_labels(model, dataloader)

            # Compute metrics
            f1 = self.compute_f1(predictions, labels)
            iou = self.compute_iou(predictions, labels)

            # Log metrics
            print(f"Step {state.global_step}: F1 Score: {f1}, IoU Score: {iou}")