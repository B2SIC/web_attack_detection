from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def filtering(raw_data):
    return raw_data.replace('\n', ' ')

def parsing(path):  # 파싱을 진행하는 함수
    with open(path, 'r', encoding = 'utf-8') as f:
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
    return train_vec, test_vec

def train(train_vec, train_y):
    rf = LogisticRegression(C=3, random_state=0, max_iter=10000)
    rf.fit(train_vec, train_y)

    # 최적의 하이퍼 파라미터 찾기
    # params = {'penalty': ['l2', 'l1'],
    #           'C': [0.01, 0.1, 1, 3, 5, 10]}
    # #
    # grid_clf = GridSearchCV(rf, param_grid=params, scoring='accuracy', cv=3)
    # grid_clf.fit(train_vec, train_y)
    # print(grid_clf.best_params_, grid_clf.best_score_)
    return rf

def test(test_y, test_vec, rf):
    pred = rf.predict(test_vec)

    print("Accuracy Score:", accuracy_score(test_y, pred))
    print("Precision Score:",precision_score(test_y, pred))
    print("Recall Score:", recall_score(test_y, pred))
    print("F1 Score:", f1_score(test_y, pred))

    return pred

train_x, train_y = dataset('./data/','train')
test_x, test_y = dataset('./data/','test')

train_vec, test_vec = vectorize(train_x, test_x)
rf = train(train_vec, train_y)
pred = test(test_y, test_vec, rf)

tf = TfidfVectorizer()
tf = tf.fit(train_x)

# print(len(tf.vocabulary_))  # 고유 단어 개수 확인
