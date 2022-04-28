import itertools
import json
import time
import pytorch_lightning as pl
import nltk
import numpy as np
import rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AdamW
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM

from heuristic_converter import get_converter
from evaluate import get_acc, get_template_acc

nltk.download("punkt")


class DS2Model(pl.LightningModule):
    def __init__(self, args, tokenizer: AutoTokenizer, summ_model: AutoModelForSeq2SeqLM, qa_model=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.summ_model = summ_model
        if self.args["use_qa_deconverter"]:
            self.qa_model = qa_model
        self.lr = args["lr"]
        self.blank = "____"

        self.converter = get_converter(args["state_converter"])
        self.evaluator = rouge.Rouge(
            metrics=["rouge-n"],
            max_n=4,
            limit_length=True,
            length_limit=100,
            length_limit_type="words",
            apply_avg=False,
            apply_best=True,
            alpha=0.5,  # default F1_score
            weight_factor=1.2,
            stemming=True,
        )

    def training_step(self, batch, batch_idx) -> dict:
        self.summ_model.train()
        outputs = self.summ_model(
            input_ids=batch["encoder_input"], attention_mask=batch["attention_mask"], labels=batch["decoder_output"],
        )

        return {"loss": outputs.loss, "log": {"train_loss": outputs.loss.detach()}}

    def eval_step(self, batch, batch_idx):
        self.summ_model.eval()
        outputs = self.summ_model(
            input_ids=batch["encoder_input"], attention_mask=batch["attention_mask"], labels=batch["decoder_output"],
        )

        return outputs.loss.item()

    def pred_step(self, batch, batch_idx) -> dict:
        self.summ_model.eval()
        pred_summary_token = self.summ_model.generate(
            batch["encoder_input"], num_beams=self.args["num_beams"], min_length=5, max_length=128, early_stopping=True,
        )

        return {
            "pred_summary_token": pred_summary_token,
            "gold_state": batch["slot_values"],
            "gold_summary": batch["output_text"],
            "eval_slots": batch["eval_slots"],
        }

    def eval_epoch_end(self, outputs):
        res = {}
        res["loss"] = np.mean(outputs)
        print(res)

        return res

    def pred_epoch_end(self, outputs: dict, mode: str = "val") -> dict:
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}
        # pred_summary (list(str)): a list of predicted summary
        pred_summary = [
            self.tokenizer.decode(_sum, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _sum in outputs["pred_summary_token"]
        ]
        if self.args["use_qa_deconverter"]:
            pred_state = self.qa_model.sum_to_state(pred_summary, outputs["eval_slots"])
        else:
            # pred_state (list(dict)): a list of a dict of dialog state from the `pred_summary`
            pred_state = [self.converter.convert_summary_to_state(_sum) for _sum in pred_summary]

        # pred_acc (dict(float)): a dict of jga, acc per slot, true acc per slot, and none acc per slot as float
        pred_acc = get_acc(pred_state=pred_state, gold_state=outputs["gold_state"], eval_slots=outputs["eval_slots"])

        # gold_templates (list(str)): a list of restored gold summary with blanks corresponding to '____'
        gold_templates = [
            self.converter.convert_state_to_summary(dialog_state=_dialog_state, is_for_template=True, blank=self.blank)
            for _dialog_state in outputs["gold_state"]
        ]
        # get acc for predicted summary and gold summary
        template_acc = get_template_acc(pred_summary=pred_summary, gold_templates=gold_templates, blank=self.blank)
        rouge_score = self.evaluator.get_scores(pred_summary, outputs["gold_summary"])["rouge-4"]["f"]
        bleu_score = [
            sentence_bleu([ref.split()], hyp.split(), smoothing_function=SmoothingFunction().method1)
            for ref, hyp in zip(outputs["gold_summary"], pred_summary)
        ]
        pred_acc.update(
            {"rouge": rouge_score, "bleu": np.mean(bleu_score), "template_acc": template_acc,}
        )

        samples = {
            "gold_summary": outputs["gold_summary"],
            "gold_state": outputs["gold_state"],
            "pred_summary": pred_summary,
            "pred_state": pred_state,
        }
        self.save_samples(samples, f'{str(pred_acc["jga"])}_{mode}')

        print(pred_acc)

        return pred_acc

    def validation_step(self, batch, batch_idx):
        """Operates on a single batch of data from the validation set."""
        if self.args["eval_loss_only"]:
            return self.eval_step(batch, batch_idx)
        else:
            return self.pred_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: list) -> dict:
        """Called at the end of the validation epoch with the outputs of all validation steps."""
        if self.args["eval_loss_only"]:
            val_result = {f"val_{k}": v for k, v in self.eval_epoch_end(outputs).items()}
        else:
            val_result = {f"val_{k}": v for k, v in self.pred_epoch_end(outputs=outputs, mode="val").items()}
        self.log_dict(val_result)

        return val_result

    def test_step(self, batch, batch_idx):
        return self.pred_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        res = {f"test_{k}": v for k, v in self.pred_epoch_end(outputs, "test").items()}
        self.log_dict(res)

        return res

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def save_samples(self, samples, name):
        if self.args["save_samples"] > 0:
            output_fields = ["only_domain", "fewshot", "grad_acc_steps", "train_batch_size", "state_converter"]
            output_name = (
                "_".join([str(self.args[k]) for k in output_fields]) + "_" + name + "_" + str(round(time.time()))
            )
            filename = f"./samples_data/{output_name}.json"
            with open(filename, "w") as f:
                json.dump({k: v[: self.args["save_samples"]] for k, v in samples.items()}, f)
