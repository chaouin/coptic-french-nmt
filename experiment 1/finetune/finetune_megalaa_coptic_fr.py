import argparse
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
)

# === Argument parser for test mode
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Quick test mode (sample + 1 epoch)")
args = parser.parse_args()

# === Enable offline mode
model_path = ""  # path redacted
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = model_path

# === Load training data
df = pd.read_csv("/home/chaouin/projects/def-richy/chaouin/data/train_segond_all_data.csv")

# === Reduce dataset size if test mode is active
if args.test:
    df = df.sample(n=100, random_state=42)
    output_dir = "test-output"
    num_epochs = 1
else:
    output_dir = "megalaa-finetuned-coptic-fr-segond-all-data"
    num_epochs = 15

# === Create Hugging Face dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# === Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
model = MarianMTModel.from_pretrained(model_path, local_files_only=True, use_safetensors=False)

# === Preprocessing function
def preprocess_function(examples):
    inputs = examples["coptic_text_romanized"]
    targets = examples["french_translation"]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# === Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    logging_dir="./logs_megalaa",
    save_strategy="epoch",
)

# === Custom callback for logging
class CustomLoggerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"\n🚀 Starting epoch {int(state.epoch) + 1}")

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"✅ Finished epoch {int(state.epoch)} - Step {state.global_step}\n")

# === Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[CustomLoggerCallback()]
)

# === Fine-tune the model
trainer.train()

# === Save the best model
trainer.save_model(f"./{output_dir}")
