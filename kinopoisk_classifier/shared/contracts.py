
MODEL_NAME = "DeepPavlov/rubert-base-cased"


LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


MAX_LENGTH = 512



HEAD_TOKENS = 256
TAIL_TOKENS = 254
