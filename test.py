import configs,os

print(configs.data.path)
print(configs.data.raw_cut)
print(os.path.join(configs.data.path, 'bpe.vocab'))

print(configs.model_path)