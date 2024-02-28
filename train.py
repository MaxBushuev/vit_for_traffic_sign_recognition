import argparse

from transformers import (ViTImageProcessor,
                          ViTForImageClassification,
                          TrainingArguments,
                          Trainer
                         )
from datasets import load_dataset
import evaluate
import torch
import numpy as np


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def main(args):
    ds = load_dataset('bazyl/GTSRB')
    labels = ds["train"].features["ClassId"].names
   
    processor = ViTImageProcessor.from_pretrained(args.model_name_or_path, return_tensors="pt")
    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def transform(batch):
        inputs = processor([x for x in batch['Path']], return_tensors='pt')

        inputs['labels'] = batch['ClassId']
        return inputs
    
    prepared_ds = ds.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels)
    )

    training_args = TrainingArguments(
        output_dir=args.model_save_dir,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=4,
        fp16=True,
        learning_rate=2e-4,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=processor,
    )

    print(len(prepared_ds["train"]))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--model_save_dir", type=str)

    args = parser.parse_args()

    main(args)
