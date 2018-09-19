from sklearn.externals import joblib


def main():

    vectorizer = joblib.load('model/C4_new_vectorizer.pkl')
    clf = joblib.load('model/C4_new_f1_SVC.pkl')

    tweet1 = 'dont put that on ur resume for a minimum wage job'
    tweet2 = 'hello world!'

    test_X = vectorizer.transform([tweet1, tweet2])
    y_pred_class = clf.predict(test_X)

    for pred in y_pred_class:
        if pred == 1.0:
            print('job')
        else:
            print('not job')

if __name__ == '__main__':
    main()