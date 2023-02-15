import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer
import torch.utils.data
from dataset.outcome import OutcomeDiagnosesDataset, collate_batch
from model.bert import BertModule
from model.proto import ProtoModule
import fire

from utils import ProjectorCallback, load_eval_buckets


def run_training(train_file,
                 val_file,
                 test_file,
                 batch_size=10,
                 gpus=1,
                 lr_prototypes=1e-3,
                 lr_others=2e-2,
                 lr_features=2e-6,
                 num_warmup_steps=100,
                 gamma = 1e-5,
                 max_length=512,
                 num_training_steps=5000,
                 check_val_every_n_epoch=1,
                 num_val_samples=None,
                 pretrained_model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 save_dir='output',
                 projector_callback=False,
                 model_type="PROTO",
                 resume_from_checkpoint=None,
                 use_sigmoid=False,
                 seed=7,
                 project_n_batches=1,
                 use_attention=True,
                 dot_product=False,
                 reduce_hidden_size=None,
                 prototype_vector_path=None,
                 attention_vector_path=None,
                 all_labels_path=None,
                 normalize=None,
                 loss="BCE",
                 final_layer=False,
                 use_prototype_loss=False,
                 eval_bucket_path=None,
                 few_shot_experiment=False):
    moco_queue = batch_size*100
    pl.utilities.seed.seed_everything(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    # 不同的experiment有不同的读取方式
    if few_shot_experiment:
        dataset = FilteredDiagnosesDataset
    else:
        dataset = OutcomeDiagnosesDataset

    train_dataset = dataset(train_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path)
    val_dataset = dataset(val_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path)
    test_dataset = dataset(test_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path)
    dataloader = {}
    for split, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
        dataloader[split] = torch.utils.data.DataLoader(dataset,
                                                        collate_fn=collate_batch,
                                                        batch_size=batch_size,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        shuffle=split == "train",
                                                        sampler=RandomSampler(dataset,
                                                                              replacement=True,
                                                                              num_samples=num_val_samples)
                                                        if split != "train" else None)

    eval_buckets = load_eval_buckets(eval_bucket_path)

    if model_type is "BERT":
        model = BertModule(pretrained_model=pretrained_model,
                           num_classes=dataset.get_num_classes(),
                           lr_features=lr_features,
                           lr_others=lr_others,
                           num_training_steps=num_training_steps,
                           num_warmup_steps=num_warmup_steps,
                           save_dir=save_dir,
                           use_attention=use_attention,
                           reduce_hidden_size=reduce_hidden_size,
                           seed=seed,
                           eval_buckets=eval_buckets
                           )

    elif model_type is "PROTO":
        model = ProtoModule(pretrained_model=pretrained_model,
                            label_order_path=all_labels_path,
                            num_classes=dataset.get_num_classes(),
                            num_training_steps=num_training_steps,
                            lr_features=lr_features,
                            lr_others=lr_others,
                            lr_prototypes=lr_prototypes,
                            use_sigmoid=use_sigmoid,
                            loss=loss,
                            final_layer=final_layer,
                            use_prototype_loss=use_prototype_loss,
                            num_warmup_steps=num_warmup_steps,
                            use_cuda=gpus > 0,
                            save_dir=save_dir,
                            use_attention=use_attention,
                            dot_product=dot_product,
                            normalize=normalize,
                            reduce_hidden_size=reduce_hidden_size,
                            prototype_vector_path=prototype_vector_path,
                            attention_vector_path=attention_vector_path,
                            seed=seed,
                            moco_queue = 300,
                            eval_buckets=eval_buckets,
                            gamma=gamma
                            )

    else:
        raise Exception(f"{model_type} not found. Please choose a valid model_type.")

    tb_logger = TensorBoardLogger(save_dir, name="lightning_logs")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val/auroc_macro',
                                          mode='max',
                                          save_last=True,
                                          save_top_k=1,
                                          dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
                                          filename='ckpt-{epoch:02d}')
    early_stop_callback = EarlyStopping(monitor="val/auroc_macro",
                                        patience=25,
                                        mode="max")

    callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]
    if projector_callback:
        embedding_projector_callback = ProjectorCallback(dataloader["train"], project_n_batches=project_n_batches)
        callbacks.append(embedding_projector_callback)

    
    trainer = pl.Trainer(callbacks=callbacks,
                         logger=tb_logger,
                         default_root_dir=save_dir,
                         gpus=gpus,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         deterministic=True,
                         accelerator="ddp",
                         precision=32,
                         resume_from_checkpoint=resume_from_checkpoint
                         )


    trainer.fit(model, dataloader["train"], dataloader["val"])
    trainer.test(dataloaders=dataloader["test"], ckpt_path="best")


if __name__ == '__main__':
    
    fire.Fire(run_training)
