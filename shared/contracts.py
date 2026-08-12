# Базовая модель
MODEL_NAME = "DeepPavlov/rubert-base-cased"

# Маппинг классов. 
LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

# Длина входа. 512 — архитектурный потолок BERT-base.
MAX_LENGTH = 512

# Head+tail обрезка: в отзывах вывод в конце, наивная обрезка "с головы"
# теряет вердикт у 36% длинных текстов. HEAD + TAIL + [CLS] + [SEP] = 512.
HEAD_TOKENS = 256
TAIL_TOKENS = 254