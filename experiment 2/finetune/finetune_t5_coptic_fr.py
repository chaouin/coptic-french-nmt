import argparse
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq
)

# === Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the training CSV file")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
parser.add_argument("--test", action="store_true", help="Quick test mode (sample + 1 epoch)")
args = parser.parse_args()

# === Model path and offline cache
model_path = ""  # path redacted
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = model_path

# === Load training data
df = pd.read_csv(args.data_path)
df.dropna(subset=["coptic_text_romanized", "french_translation"], inplace=True)

# === Reduce data if test mode is enabled
if args.test:
    df = df.sample(n=100, random_state=42)
    num_epochs = 1
    print("⚙️ Test mode enabled: 100 samples, 1 epoch")
else:
    num_epochs = 15

# === Create Hugging Face dataset
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1)

# === Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=False)

# === Preprocessing
def preprocess_function(examples):
    # For T5 models, it's common to add an explicit task prefix
    inputs = ["translate Coptic to French: " + text for text in examples["coptic_text_romanized"]]
    targets = examples["french_translation"]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === Tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# === Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    logging_dir="./logs_t5",
    save_strategy="epoch",
)

# === Custom logger callback
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

# === Fine-tuning
trainer.train()

# === Save model
trainer.save_model(f"./{args.output_dir}")
