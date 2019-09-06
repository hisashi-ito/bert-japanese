#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 【preprocess】
#
# 概要:
#      finetune-to-livedoor-corpus.ipynb を python コード化
#
import configparser
import glob
import os
import pandas as pd
import subprocess
import sys
import tarfile
from urllib.request import urlretrieve

# 設定ファイルを設定
CURDIR = os.getcwd()
CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

# 設定値をセット
FILEURL = config['FINETUNING-DATA']['FILEURL']
FILEPATH = config['FINETUNING-DATA']['FILEPATH']
EXTRACTDIR = config['FINETUNING-DATA']['TEXTDIR']

# 保存先のディレクトリがない場合は作成する
if not os.path.exists(os.path.dirname(FILEPATH)):
    os.mkdir(os.path.dirname(FILEPATH))
if not os.path.exists(os.path.dirname(EXTRACTDIR)):
    os.mkdir(os.path.dirname(EXTRACTDIR))

# Download and unzip data
def download():
    urlretrieve(FILEURL, FILEPATH)
    tar = tarfile.open(FILEPATH, "r:gz")
    tar.extractall(EXTRACTDIR)
    tar.close()

# Data preprocessing
def extract_txt(filename):
    with open(filename) as text_file:
        # 0: URL, 1: timestamp
        text = text_file.readlines()[2:]
        text = [sentence.strip() for sentence in text]
        text = list(filter(lambda line: line != '', text))
        return ''.join(text)

# corpus
def corpus():
    # テストとラベルのコーパス情報
    all_text = []
    all_label = []
    # 改行コード等を変換
    table = str.maketrans({
        '\n': '',
        '\t': '　',
        '\r': '',
    })
    # ラベルの値となるディレクトリ名(カテゴリ)を取得
    categories = [
        name for name
        in os.listdir( os.path.join(EXTRACTDIR, "text") )
        if os.path.isdir( os.path.join(EXTRACTDIR, "text", name) ) ]
    categories = sorted(categories)

    for cat in categories:
        files = glob.glob(os.path.join(EXTRACTDIR, "text", cat, "{}*.txt".format(cat)))
        files = sorted(files)
        body = [ extract_txt(elem).translate(table) for elem in files ]
        # text の数だけ .. [cat1,cat2,,,,]
        label = [cat] * len(body)
        all_text.extend(body)
        all_label.extend(label)
    df = pd.DataFrame({'text' : all_text, 'label' : all_label})
    print("*** normal ***")
    print(df.head())
    df = df.sample(frac=1, random_state=23).reset_index(drop=True)
    print("*** shuf ***")
    print(df.head())
    return df

# save
def save(df):
    # Save data as tsv files.
    # test:dev:train = 2:2:6. To check the usability of finetuning,
    # we also prepare sampled training data (1/5 of full training data).
    df[:len(df) // 5].to_csv( os.path.join(EXTRACTDIR, "test.tsv"), sep='\t', index=False)
    df[len(df) // 5:len(df)*2 // 5].to_csv( os.path.join(EXTRACTDIR, "dev.tsv"), sep='\t', index=False)
    df[len(df)*2 // 5:].to_csv( os.path.join(EXTRACTDIR, "train.tsv"), sep='\t', index=False)
    

download()    # コーパスデータをダウンロード
df = corpus() # コーパスデータをデータフレームへ変換
save(df)      # コーパスデータを保存
