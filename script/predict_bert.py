#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【predict_bert】
#
# 概要:
#      bert を分類問題に適用した場合の推論器
#
# TPU 利用時:
# https://qiita.com/uedake722/items/fb9877fc45224353b44b
#
import sys

# bert-japanese
sys.path.append("../src")
import tokenization_sentencepiece as tokenization
from run_classifier import LivedoorProcessor
from run_classifier import model_fn_builder
from run_classifier import file_based_input_fn_builder
from run_classifier import file_based_convert_examples_to_features
from utils import str_to_value

# bert
sys.path.append("../bert")
import modeling
import optimization
import tensorflow as tf
import configparser
import json
import glob
import os
import pandas as pd
import tempfile
import re

# report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 設定類
CURDIR = os.getcwd()
CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)
# 設定値をセット
FILEURL = config['FINETUNING-DATA']['FILEURL']
FILEPATH = config['FINETUNING-DATA']['FILEPATH']
EXTRACTDIR = config['FINETUNING-DATA']['TEXTDIR']
PRETRAINED_MODEL_PATH = '../model/model.ckpt-1400000'
FINETUNE_OUTPUT_DIR = '../model/livedoor_output'

class FLAGS(object):
    '''Parameters.'''
    def __init__(self, finetuned_model_path):
        # sentencepiece model
        self.model_file = "../model/wiki-ja.model"
        self.vocab_file = "../model/wiki-ja.vocab"
        self.do_lower_case = True
        self.use_tpu = False
        self.output_dir = "/dummy"
        self.data_dir = "/work/data/livedoor"
        self.max_seq_length = 512
        self.init_checkpoint = finetuned_model_path
        self.predict_batch_size = 4
        
        # The following parameters are not used in predictions.
        # Just use to create RunConfig.
        self.master = None
        self.save_checkpoints_steps = 1
        self.iterations_per_loop = 1
        self.num_tpu_cores = 1
        self.learning_rate = 0
        self.num_warmup_steps = 0
        self.num_train_steps = 0
        self.train_batch_size = 0
        self.eval_batch_size = 0

def latest_ckpt_model():
    models = {}
    for name in glob.glob("{}/model.ckpt*data*".format(FINETUNE_OUTPUT_DIR)):
        m = re.search(r'model.ckpt\-(\d+)\.data', name)
        if m:
            models[int(m.group(1))] = name
    latest_key = sorted(models, reverse=True)[0]
    return models[latest_key]

def accracy(result, label_list):
    import pandas as pd
    test_df = pd.read_csv("/work/data/livedoor/test.tsv", sep='\t')
    test_df['predict'] = [ label_list[elem['probabilities'].argmax()] for elem in result ]
    acc = sum( test_df['label'] == test_df['predict'] ) / len(test_df)

    # 正答率(accuracy)を表示
    print("*** accuracy: {} ***".format(acc))

    # 詳細なレポートが簡単に見れる
    # support は正解データの数
    print("*** classification_report ***")
    print(classification_report(test_df['label'], test_df['predict']))
    print("*** 混合行列 ***")
    print(confusion_matrix(test_df['label'], test_df['predict']))
    
def main():
    # bert
    bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
    bert_config_file.write(json.dumps({k:str_to_value(v) for k,v in config['BERT-CONFIG'].items()}))
    bert_config_file.seek(0) # [注意] 最初からread するから
    bert_config = modeling.BertConfig.from_json_file(bert_config_file.name)
    latest_ckpt = latest_ckpt_model()
    # model.ckpt-11052.index, model.ckpt-11052.meta データの prefix
    finetuned_model_path = latest_ckpt.split('.data-00000-of-00001')[0]
    flags = FLAGS(finetuned_model_path)
    processor = LivedoorProcessor()
    label_list = processor.get_labels()
    
    # sentencepiece
    tokenizer = tokenization.FullTokenizer(
        model_file=flags.model_file, vocab_file=flags.vocab_file,
        do_lower_case=flags.do_lower_case)

    # no use TPU
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # config
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=flags.master,
        model_dir=flags.output_dir,
        save_checkpoints_steps=flags.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=flags.iterations_per_loop,
            num_shards=flags.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=flags.init_checkpoint,
        learning_rate=flags.learning_rate,
        num_train_steps=flags.num_train_steps,
        num_warmup_steps=flags.num_warmup_steps,
        use_tpu=flags.use_tpu,
        use_one_hot_embeddings=flags.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=flags.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=flags.train_batch_size,
        eval_batch_size=flags.eval_batch_size,
        predict_batch_size=flags.predict_batch_size)

    # テストデータコレクションの取得
    predict_examples = processor.get_test_examples(flags.data_dir)
    predict_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.tf_record')
    """Convert a set of `InputExample`s to a TFRecord file."""
    """出力: predict_file.name """
    # https://github.com/yoheikikuta/bert-japanese/blob/master/src/run_classifier.py#L371-L380    
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            flags.max_seq_length, tokenizer,
                                            predict_file.name)
    predict_drop_remainder = True if flags.use_tpu else False

    # TPUEstimatorに渡すクロージャを作成
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file.name,
        seq_length=flags.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)
    # 推論
    result = estimator.predict(input_fn=predict_input_fn)
    result = list(result)
    
    # 精度を計算
    accracy(result, label_list)
    
if __name__ == '__main__':
    main()
