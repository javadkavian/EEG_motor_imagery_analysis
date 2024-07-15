from sklearn.metrics import classification_report
from metrics import plot_confusion_matrix, plot_roc
import matplotlib.pyplot as plt


def train_test_report(model, X_train, X_test, y_train, y_test, labels, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=labels))
    plot_confusion_matrix(y_test, y_pred, labels, model_name)
    plt.savefig('../assets/' + model_name.replace(" ", "_") + '_cm.png')
    plot_roc(y_test, y_score, model_name)
    plt.savefig('../assets/' + model_name.replace(" ", "_") + '_roc.png')
