import numpy as np 
import pandas as pd 
import matplotlib
import sqlite3
from sklearn import model_selection
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import re
import urllib
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import cross_validation, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import os
from flask import Flask, render_template, redirect, url_for, json, request
from flask import Flask, session, redirect, url_for, escape, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
#app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    session.pop('username', None)
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                session['username'] = form.username
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1>Invalid username or password</h1>'
        

    return render_template('login_form.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return '<h1>New user has been created!</h1>'
        

    return render_template('signup.html', form=form)


@app.route('/enternew')
def new_student():
   return render_template('disease.html')




@app.route("/disease")
def diseaseInput():
    return render_template('disease_form.html',name=current_user.username)    

@app.route('/diseasePredictionPage',methods=['POST'])
def diseasePredictionPage():
    # read the posted values from the UI
        _symptom1 = request.form['symptom1']
        _symptom2 = request.form['symptom2']
        _symptom3 = request.form['symptom3']
        _symptom4 = request.form['symptom4']
        _symptom5 = request.form['symptom5']
        _symptom6 = request.form['symptom6']
        _symptom7 = request.form['symptom7']
        _symptom8 = request.form['symptom8']

        data = pd.read_csv("Manual-Data/Training.csv")
        df = pd.DataFrame(data)
        cols = df.columns
        np.set_printoptions(threshold=np.inf)
        cols = cols[:-1]
        x = df[cols]
        y = df['prognosis']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44)
        dt = RandomForestClassifier(n_estimators=100) 
        clf_dt=dt.fit(x_train,y_train)
        feature_dict = {}
        features=x
        for i,f in enumerate(features):
            feature_dict[f] = i
            
        symptom = [_symptom1,_symptom2,_symptom3,_symptom4,_symptom5,_symptom6,_symptom7,_symptom8]
        for i in range(len(symptom)):
            x = feature_dict[symptom[i]]
            sample_x = [i/x if i ==x else i*0 for i in range(len(feature_dict))]
            
            
        sample_x = np.array(sample_x).reshape(1,len(sample_x))
       
        disease = dt.predict(sample_x)
        disname = disease[0]
     

        return render_template('disease_Prediction.html',value = disname)


@app.route("/diabetes")
def diabetesInput():
    return render_template('diabetes_form.html')
    

@app.route('/diabetesPredictionPage',methods=['POST'])
def diabetesPredictionPage():
    # read the posted values from the UI
  
    _glucose = request.form['glucose']
    _pressure = request.form['pressure']
    _insulin = request.form['insulin']
    _bmi = request.form['bmi']
    _age = request.form['age']
    _preg = request.form['pregnancy']

    # url with dataset
  
    # load the CSV file as a numpy matrix
    data = pd.read_csv("C:\\Users\Sampada\Desktop\myproject\app3\diabetes.csv",header=0)
    data['BMI'] = data['BMI'].astype(int)
    data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].astype(int)
    features = list(data.columns[:8])
    y = data['Outcome']
    x = data[features]
    Tree = tree.DecisionTreeClassifier()
    Tree = Tree.fit(x,y)
   
    X_test=[_preg,_glucose,_pressure,0,_insulin,_bmi,0.134,_age]

    diab_dt = DecisionTreeClassifier().fit(x, y)
    y_pred = diab_dt.predict(X_test)

    output= int(y_pred[0])
    if output == 0:
        return render_template('diabetes_prediction.html',value ="Congratulations!! You are safe from diabetes.Keep visiting nearby hospitals for regular checkups.")
    else:
        return render_template('diabetes_prediction.html',value ="There are high chances of having diabetes. Please visit nearby hospital soon.")

@app.route("/heartDisease")
def heartDiseaseInput():
    return render_template('heart_form.html')

@app.route('/heartDiseasePredictionPage',methods=['POST'])
def heartDiseasePredictionPage():
    # read the posted values from the UI
    _age = request.form['age']
    _sex = request.form['sex']
    _cpt = request.form['chest_pain_types']
    _bp = request.form['resting_BP']
    _cholesterol = request.form['serum_cholesterol']
    _sugar = request.form['bloodSugar']
    _restEcg = request.form['restEcg']
    _maxHeartRate = request.form['maxHeartRate']

    data = pd.read_csv("heart_disease.csv",header=0)
    features=list(data.columns[0:13])
    train, test = train_test_split(data, test_size = 0.1)
    X_train = train[features]
    y_train = train.outcome
    X_test=[_age,_sex,_cpt,_bp,_cholesterol,_sugar,_restEcg,_maxHeartRate,0,2.5,3,0,6]
    print("input")
    print(X_test)
    heart_dt = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    y_pred = heart_dt.predict(X_test)
    output= int(y_pred[0])

    if output == 0:
        return render_template('heart_prediction.html',value ="Congratulations!! You are safe from heart disease.Keep visiting nearby hospitals for regular checkups.")
    else:
        return render_template('heart_prediction.html',value ="There are high chances of having heart disease. Please visit nearby hospital soon.")

@app.route("/liverDiseaseInput")
def liverDiseaseInput():
    return render_template('liver_form.html')

@app.route('/liverDiseasePredictionPage',methods=['POST'])
def liverDiseasePredictionPage():
    # read the posted values from the UI
    _Age = request.form['Age']
    _Total_Bilirubin = request.form['Total_Bilirubin']
    _Direct_Bilirubin = request.form['Direct_Bilirubin']
    _Alkaline_Phosphotase = request.form['Alkaline_Phosphotase']
    _Alamine_Aminotransferase = request.form['Alamine_Aminotransferase']
    _Aspartate_Aminotransferase = request.form['Aspartate_Aminotransferase']
    _Total_Protiens = request.form['Total_Protiens']
    _Albumin = request.form['Albumin']
    _Albumin_and_Globulin_Ratio = request.form['Albumin_and_Globulin_Ratio']
    _Female = request.form['Gender']
    _Male = request.form['Gender']



    df = pd.read_csv('indian_liver_patient.csv')
    df = pd.concat([df,pd.get_dummies(df['Gender'], prefix = 'Gender')], axis=1)
    df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())
    x = df.drop(['Gender','Dataset'], axis=1)
    features = x
    y = df['Dataset']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    X_test = [_Age,_Total_Bilirubin,_Direct_Bilirubin,_Alkaline_Phosphotase,_Alamine_Aminotransferase,_Aspartate_Aminotransferase,_Total_Protiens,_Albumin,_Albumin_and_Globulin_Ratio,1,0]

    dtree_predicted = decision_tree.predict(X_test)
    output= int(dtree_predicted[0])
    if output == 1:
        return render_template('heart_prediction.html',value ="Congratulations!! You are safe from liver disease.Keep visiting nearby hospitals for regular checkups.")
    else:
        return render_template('heart_prediction.html',value ="There are high chances of having liver disease. Please visit nearby hospital soon.")

@app.route('/dashboard')
@login_required
def dashboard():
     if 'username' in session:
      username = session['username']
      return render_template('dash_index1.html', name=current_user.username)
    


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
