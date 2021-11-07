import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score

def filtering(raw_data):
    return raw_data.replace('\n', ' ')

def parsing(path):  # 파싱을 진행하는 함수
    with open(path, 'r', encoding = 'utf-8') as f:  # 파일을 읽어들이고 ['로그','로그',...] 이런식으로 로그를 구조화
        train = []
        para = ""
        filter_tuple = ('GET', 'POST', 'PUT')
        while True:
            l = f.readline() # 한줄씩 읽어 옵니다

            if not l:
                break # 파일을 전부 읽으면 읽기를 중단합니다.

            if l != "\n":
                if l.startswith(filter_tuple):
                    para += l
            else:
                if para != '':
                    # if para[:4] == 'POST' or para[:3] == 'PUT': # Method가 POST인 경우 예외적으로 바디까지 가져옵니다.
                    if para.startswith(('POST', 'PUT')):
                        para += f.readline()
                    train.append(filtering(para))
                    para = ""
    return train

def dataset(path, mod='train'): # 데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다
    # 라벨에서는 0이 정상, 1이 비정상
    x = parsing(f'{path}norm_{mod}.txt') # mod에 따라 train을 가져올지 test 데이터를 가져올지 결정됩니다.
    y = [0] * len(x) # 정상 라벨 0 을 정상 데이터 개수 만큼 생성

    x += parsing(f'{path}anomal_{mod}.txt')
    y += [1] * (len(x)-len(y)) # 비정상 라벨 1을 비정상 데이터 개수 만큼 생성
    return x, y

def vectorize(train_x, test_x): # 문장을 벡터로 만듭니다 해당 코드에서는 기본적인 tf idf를 사용하고 있습니다.
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)
    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)
    return train_vec,test_vec

def train(train_vec, train_y): # 랜덤 포레스트로 훈련 시킵니다. 모델을 바꾸고 싶다면 이 함수를 변경해야 합니다.
    # rf = RandomForestClassifier()
    rf = LogisticRegression(random_state=0, max_iter=10000)
    rf.fit(train_vec, train_y)
    return rf

def test(test_y, test_vec, rf): #입력 받은 테스트와 모델로 테스트를 실시합니다
    pred = rf.predict(test_vec)

    print("Accuracy Score:", accuracy_score(test_y, pred))  # Accuracy는 올바르게 예측된 데이터의 수를 전체 데이터로 나눈 값
    print("Precision Score:",precision_score(test_y, pred))
    print("Recall Score:", recall_score(test_y, pred))
    print("F1 Score:", f1_score(test_y, pred))  # F1 Score는 Precision과 Recall의 조화평균 값

    return pred

train_x, train_y = dataset('./data/','train')
test_x, test_y = dataset('./data/','test')

train_vec, test_vec = vectorize(train_x, test_x)
rf = train(train_vec, train_y)
# rf = train(test_vec, test_y)
pred = test(test_y, test_vec, rf)

tf = TfidfVectorizer()
tf = tf.fit(train_x)

print(len(tf.vocabulary_))  # 고유 단어 개수 확인
