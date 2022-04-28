from asyncio.log import logger
import os

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM

from dataloader import prepare_data
from config import get_args
from model import DS2Model


def fine_tune(args):
    args = vars(args)
    seed_everything(args["seed"])
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args["model_checkpoint"])
    summ_model = AutoModelForSeq2SeqLM.from_pretrained(args["model_checkpoint"])

    # get dataloader as dict format
    dataloaders, _ = prepare_data(args, tokenizer)
    print("Created dataloaders")

    # settings for logging
    exp_name = args["exp_name"]
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    log_path = f"./logs/{exp_name}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if args["load_pretrained"]:
        pretrain_ckpt_path = os.path.join(args["load_pretrained"], "pretrain")
        pretrain_ckpts = [_ckpt for _ckpt in os.listdir(pretrain_ckpt_path) if ".ckpt" in _ckpt]
        assert len(pretrain_ckpts) == 1
        ckpt = pretrain_ckpts[0]
        print("Load pre-trained model from: ", os.path.join(pretrain_ckpt_path, ckpt))
        dst_model = DS2Model.load_from_checkpoint(
            checkpoint_path=os.path.join(pretrain_ckpt_path, ckpt),
            args=args,
            tokenizer=tokenizer,
            sum_model=summ_model,
            qa_model=None,
        )
    else:
        dst_model = DS2Model(args=args, tokenizer=tokenizer, summ_model=summ_model, qa_model=None)

    print("Created Model")

    # set dir for .log file
    dir_path = os.path.join(log_path, args["mode"])
    if not args["do_test_only"]:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss" if args["eval_loss_only"] else "val_jga",
            patience=args["patience"],
            verbose=False,
            mode="min" if args["eval_loss_only"] else "max",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,
            filename="{val_loss:.4f}" if args["eval_loss_only"] else "{val_jga:.4f}",
            monitor="val_loss" if args["eval_loss_only"] else "val_jga",
            save_top_k=3,
            mode="min" if args["eval_loss_only"] else "max",
        )
        callbacks = [earlystopping_callback, checkpoint_callback]
    else:
        callbacks = None

    # profiler = PyTorchProfiler(export_to_chrome=True)
    trainer = Trainer(
        callbacks=callbacks,
        accumulate_grad_batches=args["grad_acc_steps"],
        gradient_clip_val=args["max_norm"],
        max_epochs=args["num_epochs"],
        gpus=args["num_gpus"],
        deterministic=True,  # whether PyTorch operations must use deterministic algorithms
        devices=args["num_gpus"],
        accelerator="gpu",
        strategy="ddp",
        val_check_interval=args["val_check_interval"],
        logger=CSVLogger(dir_path, f"seed_{args['seed']}") if not args["do_test_only"] else None,
        resume_from_checkpoint=args["resume_from_ckpt"],
    )

    if not args["do_test_only"]:
        trainer.fit(dst_model, dataloaders["train"], dataloaders["dev"])

    if not args["do_train_only"]:
        print("Start test")
        # evaluate model
        args["num_beams"] = args["test_num_beams"]
        if args["do_test_only"]:
            ckpts = [_ckpt for _ckpt in os.listdir(dir_path) if ".ckpt" in _ckpt]
            assert len(ckpts) == 1
            ckpt = ckpts[0]
            print("Load pretrained model from: ", os.path.join(dir_path, ckpt))
            ckpt_path = os.path.join(dir_path, ckpt)
        else:
            ckpt_path = checkpoint_callback.best_model_path

        dst_model = DS2Model.load_from_checkpoint(
            checkpoint_path=ckpt_path, args=args, tokenizer=tokenizer, summ_model=summ_model, qa_model=None
        )
        trainer.test(dst_model, dataloaders["test"])


if __name__ == "__main__":
    args = get_args()
    fine_tune(args)
