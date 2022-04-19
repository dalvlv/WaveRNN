from pypinyin import pinyin, Style, slug
import re

def text_to_pinyin(text):
    '''中文转拼音'''
    text_pinyin = pinyin(text, style=Style.TONE3)
    new_text = ''
    for i in text_pinyin:
        i2 = i[0]
        if re.match('[a-z]', i2[-1]):
            i2 = i2 + '5'

        new_text = new_text + ' ' + i2

    new_text = new_text.lstrip()
    return new_text
