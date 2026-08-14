from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("seara/rubert-base-cased-russian-sentiment")
print("id2label:", cfg.id2label)
print("num_labels:", cfg.num_labels)
print("max_position_embeddings:", cfg.max_position_embeddings)
