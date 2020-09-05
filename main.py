#from app import app
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def index():
    data = pd.read_csv('glass.csv')

    array = data.values
    X = array[:, 0:9]
    Y = array[:, 9]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30)

    classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 20), max_iter=2000, activation='relu', solver='adam',
                               random_state=1)

    classifier.fit(X_train, Y_train)
    filename = 'glass.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))

    y_pred = loaded_model.predict(X_test)

    labels = np.unique(y_pred)

    accuracy = accuracy_score(Y_test, y_pred)

    precision = precision_score(Y_test, y_pred, average='weighted', labels=labels)

    recall = recall_score(Y_test, y_pred, average='weighted', labels=labels)

    score = f1_score(Y_test, y_pred, average='weighted', labels=labels)

    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=labels).ravel()
    print("True negatif= {}, False Positif = {}, False Negatif = {} , and True Positif = {} ".format(tn, fp, fn, tp))
    # print("Banyak data Train = {}".format(len(X_train)))
    # print("Banyak data Train = {}".format(len(Y_train)))
    # print("Banyak data Train = {}".format(len(X_test)))
    # print("Banyak data Train = {}".format(len(Y_test)))
    # print("Banyak data Train = {}".format(len(y_pred)))
    con_mat = pd.crosstab(Y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    plt.title('Confusion matrix of the classifier')
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(con_mat, annot=True, annot_kws={"size": 20})  # font size
    plt.savefig('static/images/plot.png')

    if request.method == 'POST':
      accuracy = accuracy
      recall = recall
      precision = precision
      f1 = score
      return redirect(url_for("result", accuracy=accuracy, recall=recall, precision=precision, f1=f1))
    else:
        return render_template('index.html', title='Machine Learning')

@app.route('/<accuracy>/<recall>/<precision>/<f1>')
def result(accuracy, recall, precision, f1):
    return render_template('index.html', title='Machine Learning', accuracy=accuracy, recall=recall, precision=precision, f1=f1, url='/static/images/plot.png')

@app.route('/result_pred',methods = ['POST', 'GET'])
def result_pred():
    data = pd.read_csv('glass.csv')

    array = data.values
    X = array[:, 0:9]
    Y = array[:, 9]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30)

    classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 20), max_iter=2000, activation='relu', solver='adam',
                               random_state=1)

    classifier.fit(X_train, Y_train)
    filename = 'glass.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))

    if request.method == 'POST':
      RI = request.form.get('ri', type=float)
      Na = request.form.get('na', type=float)
      Mg = request.form.get('mg', type=float)
      Al = request.form.get('al', type=float)
      Si = request.form.get('si', type=float)
      K = request.form.get('k', type=float)
      Ca = request.form.get('ca', type=float)
      Ba = request.form.get('ba', type=float)
      Fe = request.form.get('fe', type=float)
      test = [[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]]
      # print(test)
      result = loaded_model.predict(test)
      return render_template('index.html', title='Machine Learning', result=str(result))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
