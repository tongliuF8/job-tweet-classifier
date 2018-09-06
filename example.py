from sklearn.externals import joblib
import timeit, math


def main():
    # mark the beginning time of process
    start = timeit.default_timer()

    vectorizer = joblib.load('vectorizer.pkl')
    clf = joblib.load('f1_SVC.pkl')

    message = 'dont put that on ur resume for a minimum wage job'

    test_X = vectorizer.transform([message])
    y_pred_class = clf.predict(test_X)[0]

    print(y_pred_class)

    ##### mark the ending time of process #####
    end = timeit.default_timer()
    seconds = math.ceil(end - start)
    # Convert Secs Into Human Readable Time String (HH:MM:SS)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print "This process took %d:%02d:%02d" % (h, m, s)


if __name__ == '__main__':
    main()