import configs
from transformers import GPT2LMHeadModel, BertTokenizer, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import sentencepiece as spm

model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(f"{configs.data.path}spiece.model")
tokenizer.load_vocabulary(f"{configs.data.path}spiece.vocab",threshold=0)


dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=f"{configs.data.raw}",
    block_size=128  # 指定文本块的大小
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(f"{configs.data.model_path}")

