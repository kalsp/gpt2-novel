import configs
import sentencepiece as spm

# 加载 Tokenizer 模型
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(f"{configs.data.path}spiece.model")
tokenizer.load_vocabulary(f"{configs.data.path}spiece.vocab",threshold=0)

# 使用 Tokenizer 进行分词
text = "浑浑噩噩 之间 ， 他 觉得 心中 憋 得 难受 ， 忍不住 发出 一声 声响 ， 手臂 也 轻轻 抬起"
tokens = tokenizer.encode_as_pieces(text)
print(tokens)