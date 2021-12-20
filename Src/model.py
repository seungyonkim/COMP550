import os
import os.path as osp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
from konlpy.tag import Okt # tweepy must be downgraded to version 3.10.0 'pip install tweepy == 3.10.0'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

#global variables
curdir = os.getcwd()
train_path = osp.join('..','Data', 'train.csv')
test_path = osp.join('..','Data', 'test.csv')
test_answer_path = osp.join('..', 'Data', 'test_answer.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_ans_df = pd.read_csv(test_answer_path)

cols_target = ['obscene','insult','toxic','identity_hate','threat']

stop_words_path = osp.join('..','Data', 'korean_stopwords.txt')
stop_words = pd.read_csv(stop_words_path)
stop_words_lst = []
for idx, row in stop_words.iterrows():
    stop_words_lst.append(row['stopwords'])


train_df['toxic'] = train_df['toxic'].fillna(0)
train_df['obscene'] = train_df['obscene'].fillna(0)
train_df['threat'] = train_df['threat'].fillna(0)
train_df['insult'] = train_df['insult'].fillna(0)
train_df['identity_hate'] = train_df['identity_hate'].fillna(0)


train_df['toxic'] = train_df['toxic'].astype(int)
train_df['obscene'] = train_df['obscene'].astype(int)
train_df['threat'] = train_df['threat'].astype(int)
train_df['insult'] = train_df['insult'].astype(int)
train_df['identity_hate'] = train_df['identity_hate'].astype(int)

test_df = test_df.drop(columns=['toxic','obscene','threat','insult','identity_hate'])






# check missing values in numeric columns
def check_missing_val (train_df):
    unlabelled_in_all = train_df[(train_df['toxic']!=1) & (train_df['obscene']!=1) &
                                (train_df['threat']!=1) & (train_df['insult']!=1) & (train_df['identity_hate']!=1)]

    return 'Percentage of unlabelled comments is ', len(unlabelled_in_all)/len(train_df)*100


# check for any 'null' comment
def check_null(train_df):
    no_comment = train_df[train_df['text'].isnull()]
    return len(no_comment)


def visualization (train_df,test_df):

    train_df['char_length'] = train_df['text'].apply(lambda x: len(str(x)))

    sns.set()
    train_df['char_length'].hist()
    plt.show()

    data = train_df[cols_target]

    colormap = plt.cm.plasma
    plt.figure(figsize=(7,7))
    plt.title('Correlation of features & targets',y=1.05,size=14)
    sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
               linecolor='white',annot=True)

    test_df['char_length'] = test_df['text'].apply(lambda x: len(str(x)))

    plt.figure()
    plt.hist(test_df['char_length'])
    plt.show()


def clean_text(text):
    text = re.sub('\W', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]
def clean_comment_text (train_df, test_df):
    train_df['text'] = train_df['text'].map(lambda com : clean_text(com))

    # clean the comment_text in test_df [Thanks, Pulkit Jha.]
    test_df['text'] = test_df['text'].map(lambda com : clean_text(com))

    X = train_df.text
    test_X = test_df.text

    return X, test_X



X, test_X = clean_comment_text(train_df,test_df)

def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))

#################### train ##################

def tokenized_train():
    tokenized = []
    total_tokenized = []
    #tokenized form
    for i in range(0,len(X)):
        okt_pos = Okt().pos(X[i], norm=True, stem=True)
        # okt_pos is  list with tokenizd words and corresponding grammar tag.

        okt_filtering = [x for x, y in okt_pos if y in ['Noun','Adjective', 'Verb']]


        # remove stopwords
        for i in okt_filtering:
            if i in stop_words_lst:
                okt_filtering.remove(i)

        total_tokenized.append(okt_filtering)

    #from a list to test form which is a list to string.

    # helper method to append strings to one string with a space in between


    test_form_tokenized = []

    for i in total_tokenized:
        str = listToString(i)
        test_form_tokenized.append(str)


    # change the train_df['text] to test_form_tokenized

    for idx, row in train_df.iterrows():
        train_df.at[idx,'text'] = test_form_tokenized[idx]

    return train_df

#################### test ##################

def tokenized_test():
    tokenized_test = []
    total_tokenized_test = []

    #tokenized form
    for i in range(0,len(test_X)):
        okt_pos = Okt().pos(test_X[i], norm=True, stem=True)
        # okt_pos is  list with tokenizd words and corresponding grammar tag.


        okt_filtering_test = [x for x, y in okt_pos if y in ['Noun','Adjective', 'Verb']]

        # remove stopwords
        for i in okt_filtering_test:
            if i in stop_words_lst:
                okt_filtering_test.remove(i)


        total_tokenized_test.append(okt_filtering_test)


    test_form_tokenized_test = []

    for i in total_tokenized_test:
        str = listToString(i)
        test_form_tokenized_test.append(str)


    # change the train_df['text] to test_form_tokenized


    for idx, row in train_df.iterrows():
        test_df.at[idx,'text'] = test_form_tokenized_test[idx]

    return test_df


train_df = tokenized_train()
test_df = tokenized_test()


def model(train_df, test_df):
    vect = TfidfVectorizer(max_features=5000,stop_words='english')


    X_dtm = vect.fit_transform(test_df['text'])
    test_X_dtm = vect.transform(test_X)


    # import and instantiate the Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    logreg = LogisticRegression(C=12.0)

    # create submission file
    result = pd.read_csv('../Data/final_result_template.csv')

    for label in cols_target:
        print('... Processing {}'.format(label))
        y = train_df[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm, y)
        # compute the training accuracy
        y_pred_X = logreg.predict(X_dtm)
        print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        # compute the predicted probabilities for X_test_dtm
        test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

        result[label] = test_y_prob
        try:
            result[label] = test_y_prob
        except:
            result[label] = 0
        print(label, result[label])

    return result

def token_accuracy():
    acc_list = []
    list_test = []
    list_result = []

    test_answer_path = osp.join('..', 'Data', 'result_with_tokenized.csv')
    result_df = pd.read_csv(test_answer_path)

    for idx, row in test_ans_df.iterrows():
        list_test.append(
            [int(row['toxic']), int(row['obscene']), int(row['threat']), int(row['insult']), int(row['identity_hate'])])

    for idx, row in result_df.iterrows():
        list_result.append(
            [int(row['toxic']), int(row['obscene']), int(row['threat']), int(row['insult']), int(row['identity_hate'])])

    for i, j in zip(list_test, list_result):
        # print(i,j)

        acc_list.append(accuracy_score(i, j))

    acc = sum(acc_list) / len(acc_list)

    return acc


def not_token_accuracy():
    acc_list = []
    list_test = []
    list_result = []

    test_answer_path = osp.join('..', 'Data', 'result_without_tokenized.csv')
    result_df = pd.read_csv(test_answer_path)

    for idx, row in test_ans_df.iterrows():
        list_test.append([row['toxic'], row['obscene'], row['threat'], row['insult'], row['identity_hate']])

    for idx, row in result_df.iterrows():
        list_result.append([row['toxic'], row['obscene'], row['threat'], row['insult'], row['identity_hate']])


    for test,res in zip(list_test,list_result):
        for i,j in zip(res,test):
            try:
                avg = (i / j) * 100
                acc_list.append(avg)
            except:
                avg = (i / 1) * 100
                avg = 100 - avg
                acc_list.append(avg)

    acc = sum(acc_list) / len(acc_list)

    return acc



result = model(train_df, test_df)
print(result.head(15))
#print(accuracy(result))
print("The accuracy of the untokenized model is: " + str((not_token_accuracy())))
print("The accuracy of the tokenized model is: " + str((token_accuracy())*100))
result_path = osp.join('..','Data', 'result_with_tokenized.csv')
result.to_csv(result_path,index=False)


