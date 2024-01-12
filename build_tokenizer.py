from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
import configs
import os
import shutil


tokenizer = Tokenizer(BPE())
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# def main():
#     tokenizer.train(trainer, [configs.data.raw_cut])
#     #tokenizer.save(os.path.join(configs.data.path, 'bpe.vocab'))
#     tokenizer.save(f"{configs.data.vocab}")
#     print(f"save to {configs.data.path}")


def train_with_sentenceprices(vocab_size: int = 50257, num_threads=2, character_coverage=0.98):
    os.system(f"spm_train --input={configs.data.raw_cut} --model_prefix=spiece --model_type=bpe --character_coverage={character_coverage} --vocab_size={vocab_size} --num_threads={num_threads}")
    #os.system(f"mv spiece.model {configs.data.path}")
    shutil.move("spiece.model",f"{configs.data.path}spiece.model")
    shutil.move("spiece.vocab",f"{configs.data.path}spiece.vocab")


if __name__ == '__main__':
    train_with_sentenceprices()