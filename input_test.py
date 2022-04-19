from utils.text import text_to_sequence
from tts_front_end import text_to_pinyin
inputs = []
with open('sentences.txt') as f:
    for l in f:
        l2 = text_to_pinyin(l.strip())
        seq = text_to_sequence(l2, ['basic_cleaners'])
        inputs.append(seq)
print(inputs)
