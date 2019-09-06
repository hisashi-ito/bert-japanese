#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【predict_gbdt】
#
# 概要:
#      sklearn を利用してジャンル判定を実施する
#      neologd を指定するとMeCabの辞書をipadic(DEFAULT)
#      ではなくneologd で形態素する
# 
# usage: predict_gbdt.py [--neologd]
#
# 更新履歴:
#          2019.09.06 新規作成
# 
import argparse
import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
WORK="/work/data"
NEOLOGD="/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

def main():
    parser = argparse.ArgumentParser(description='predict_gbdt')
    parser.add_argument('--neologd', action='store_true')
    args = parser.parse_args()
    train_df = pd.read_csv("{}/livedoor/train.tsv".format(WORK), sep='\t')
    dev_df = pd.read_csv("{}/livedoor/dev.tsv".format(WORK), sep='\t')
    test_df = pd.read_csv("{}/livedoor/test.tsv".format(WORK), sep='\t')
    # train, dev をどちらとも学習に利用
    train_dev_df = pd.concat([train_df, dev_df])

    # tokenizer
    if not args.neologd:
        m = MeCab.Tagger("-Owakati")
    else:
        m =  MeCab.Tagger("-Owakati -d {}".format(NEOLOGD))
    # 空打ちしておかないとたまにコケる
    m.parse("")

    # train and test
    train_dev_xs = train_dev_df['text'].apply(lambda x: m.parse(x))
    train_dev_ys = train_dev_df['label']
    test_xs = test_df['text'].apply(lambda x: m.parse(x))
    test_ys = test_df['label']
    
    vectorizer = TfidfVectorizer(max_features=750) # bertに合わせている？
    train_dev_xs_ = vectorizer.fit_transform(train_dev_xs)
    test_xs_ = vectorizer.transform(test_xs)

    # trainning
    model = GradientBoostingClassifier(n_estimators=200,
                                       validation_fraction=len(dev_df)/len(train_df),
                                       n_iter_no_change=5,
                                       tol=0.01,
                                       random_state=23)
    model.fit(train_dev_xs_, train_dev_ys)

    # accuracy report
    print("*** classification report ***")
    print(classification_report(test_ys, model.predict(test_xs_)))
    print("*** 混合行列 ***")
    print(confusion_matrix(test_ys, model.predict(test_xs_)))
    
if __name__ == "__main__":
    main()
    
