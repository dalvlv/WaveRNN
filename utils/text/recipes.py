from utils.files import get_files
from pathlib import Path
from typing import Union
import re


def ljspeech(path: Union[str, Path]):
    csv_file = get_files(path, extension='.csv')

    assert len(csv_file) == 1

    text_dict = {}

    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[-1]

    return text_dict


def biaobei(path):
    text_path = path + 'ProsodyLabeling/biaobei_pinyin.txt'
    text_dict = {}
    with open(text_path) as f:
        text = f.readlines()
        for i in range(0, len(text), 2):
            id_with_hanzi = text[i].strip().split('\t')
            id_wav = id_with_hanzi[0]
            hanzi_line = id_with_hanzi[1]
            text_line = text[i + 1].strip()
            text_line = add_biaodian(hanzi_line, text_line)
            text_dict[id_wav] = text_line
    return text_dict


def add_biaodian(hanzi, pinyin):
    # 正则表达式去除韵律标注
    hanzi = re.sub(r'#[0-9]', '', hanzi)
    # 去除单双引号
    hanzi = re.sub(r'(“|”|‘|’)', '', hanzi)
    # 去掉括号
    hanzi = re.sub(r'(（|）)', '', hanzi)
    # 获取拼音空格索引
    index_whitespace = []
    for i, letter in enumerate(pinyin):
        if letter == ' ':
            index_whitespace.append(i)
    # 提取汉字中的字符
    new_pinyin = pinyin  # 如果没有标点，返回原拼音
    j = 0  # 多个标点会改变字符对应的空格索引
    for index_char, value_char in enumerate(hanzi[:-1]):
        if value_char in ['，', '。', '！', '？', '、']:
            # 将汉字中的标点插入拼音中
            new_pinyin = new_pinyin[:index_whitespace[index_char - j - 1]] + value_char + \
                         new_pinyin[index_whitespace[index_char - j - 1]:]
            # 每增加一个标点，空格的索引应该+1
            index_whitespace = [s + 1 for s in index_whitespace]
            j += 1
    # 加入句末标点（每一句末必定有标点）
    new_pinyin = new_pinyin + hanzi[-1]

    return new_pinyin


if __name__ == '__main__':
    path = '/Users/lihaoqi/Desktop/BZNSYP/'
    text_dict = biaobei(path)
    print(text_dict)