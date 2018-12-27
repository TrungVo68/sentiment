from flask import Flask, render_template, jsonify, flash, redirect, url_for, session, request
from passlib.hash import sha256_crypt
from functools import wraps
import pandas as pd
import numpy as np
import json
import textwrap
import datetime
import flask
import flask_login

# self-defined functions for analyses
import functions as fn

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'autoencoder'

# import os
# from bs4 import BeautifulSoup
# import re
from keras.models import load_model
import pickle

path_model = "G:\\Trung\\Sentiment\\model\\"
sentiment_model = load_model(path_model+'sentiment_2channels_bidirectionalLSTM1.h5')
sentiment_tokenizer = pickle.load(open(path_model+'sentiment_tokenizer.sav', 'rb'))


# time out in 20 minutes
@app.before_request
def before_request():
    flask.session.permanent = True
    app.permanent_session_lifetime = datetime.timedelta(minutes=20)
    flask.session.modified = True
    flask.g.user = flask_login.current_user


# Home page
@app.route('/')
def index():
    return render_template('home.html')


# About
@app.route('/about')
def about():
    return render_template('about.html')


# Manual
@app.route('/manual')
def manual():
    return render_template('manual.html')


# My Sentiment
@app.route('/test')
# @is_logged_in
def test():
    return render_template('test.html')


# User Register - currently only by invitation
@app.route('/register')
def register():
    return render_template('register.html')


# Create login accounts
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Hashed password
        # Use sha256_crypt.hash() to obtain hashed password
        password = '$5$rounds=535000$noLgMo1BiU7bxLoN$ULyPBIR.' \
                   '5Vl/GLRdx6CQdfzuf5omAV0c95E/jRj08U/'

        pw_guest1 = '$5$rounds=535000$ThHPDsh.tHtKeGKk$NiMTr' \
                    'SvjfVYp0jFUD/LFS.Sjq.ZyGTOqQlYcEnfKs52'

        pw_guest2 = '$5$rounds=535000$jkoJubv/Y.CriIQf$wyoa' \
                    'PrBB0syCXD1dV.cmY2RHx5T7GsPwFLeZRRCj7k6'

        # Verify login account and password
        if username == 'admin':
            pw = password
        elif username == 'social':
            pw = pw_guest1
        elif username == 'socialengagement':
            pw = pw_guest2
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

        if sha256_crypt.verify(password_candidate, pw):
            # Passed
            session['logged_in'] = True
            session['username'] = username

            flash('You are now logged in', 'success')
            return redirect(url_for('index'))
        else:
            error = 'Invalid login'
            return render_template('login.html', error=error)

    return render_template('login.html')


# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized access, please login', 'danger')
            return redirect(url_for('login'))

    return wrap


# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


# Dashboard
@app.route('/dashboard')
@is_logged_in
def dashboard():
    return render_template('dashboard.html')


# Twitter
@app.route('/twitter')
@is_logged_in
def twitter():
    return render_template('twitter.html')


# Facebook
@app.route('/facebook')
@is_logged_in
def facebook():
    return render_template('facebook.html')


# OzBargain
@app.route('/ozbargain')
@is_logged_in
def ozbargain():
    return render_template('ozbargain.html')


# MY SENTIMENT ANALYSIS
@app.route('/_sentiment_calc')
def sentiment_calc():
    st = request.args.get('st', type=str)
    sentiment_model = load_model(path_model + 'sentiment_2channels_bidirectionalLSTM1.h5')
    output = fn.my_sentiment(st, sentiment_model, sentiment_tokenizer)
    out_message = "The sentiment of the text is {}. 1 is 100% positive and 0 is 100% negative.".format(output)
    # if output >= 0.5:
    #     out_message = "The sentiment of the text is {}% positive".format(round(output*100,2))
    # else:
    #     out_message = "The sentiment of the text is {}% negative".format(round((1-output)*100, 2))

    return jsonify(msgs=out_message)


# Twitter analysis results
@app.route('/_get_graph')
def get_graph():
    api = fn.TwitterAPILogin(AppOnly=True)

    kw = request.args.get('kw', type=str)
    mt = request.args.get('mt', type=int)
    sm = request.args.get('sm', type=str)

    # if not logged as admin, then Google API is disabled
    if sm.lower() == 'true' or session['username'] not in ['admin', 'social']:
        google = False
        print('Google NLP not in use')
    else:
        google = True
        print('Google NLP in use')

    tweets = fn.GetTweets(api=api, searchQuery=kw, maxTweets=mt)

    results, msgs = fn.AnalyseTweets(tweets=tweets, filter_by=None, Google=google)
    msgs = ' '.join(msgs)

    data = pd.DataFrame(results).transpose()

    data.columns = ['txt', 'emoji_scores', 'emoji_extracted', 'google_scores',
                    'google_magnitude', 'textblob_scores', 'vader_scores', 'ww_scores', 'textacy_scores',
                    'created_at', 'hashtags', 'user_mentions', 'in_reply_to_status_id',
                    'favorite_count', 'retweet_count', 'place', 'location',
                    'favourites_count', 'followers_count', 'friends_count',
                    'listed_count', 'statuses_count', 'numbers_list']

    # if google:
    #     data['scores'] = ((data.google_scores * data.google_magnitude
    #                        + data.textblob_scores + data.vader_scores)
    #                       / 3).astype(np.double).round(2)
    # else:
    #     data['scores'] = ((data.textblob_scores + data.vader_scores)
    #                       / 2).astype(np.double).round(2)

    if google:
        data['scores'] = (data.google_scores * data.google_magnitude).astype(np.double).round(2)
    else:
        data['scores'] = ((data.textblob_scores + data.vader_scores)
                          / 2).astype(np.double).round(2)

    data['ww_scores'] = data['ww_scores'].astype(np.double).round(2)
    avg_scores = data['scores'].mean()
    ww_avg_scores = data['ww_scores'].mean()
    msgs = msgs + ' The average sentiment score is {} for WX Model; {} for other models'.\
        format(round(ww_avg_scores, 2), round(avg_scores, 2))

    my_table = data[['txt', 'created_at', 'scores', 'ww_scores', 'favorite_count', 'retweet_count', 'location']]
    my_table['created_at'] = pd.to_datetime(my_table.created_at).dt.strftime('%Y/%m/%d %H:%M')

    df = data[['txt', 'scores', 'ww_scores', 'created_at']]
    df['key'] = pd.to_datetime(df.created_at).dt.strftime('%Y/%m/%d %H:%M')

    raw_text_list = df.txt.tolist()
    label = []
    for item in raw_text_list:
        label.append("\n".join(textwrap.wrap(item, width=30)))

    df['label'] = label
    df = df[['key', 'scores', 'ww_scores', 'label']]

    return jsonify(data=df.values.tolist(),
                   msgs=msgs,
                   my_table=json.loads(my_table.to_json(orient="split"))["data"])


# Facebook analysis results
@app.route('/_get_graph_fb')
def get_graph_fb():
    graph = fn.FacebookAPILogin(API_version='2.7')

    kw = request.args.get('kw', type=str)
    mt = request.args.get('mt', type=int)
    ft = request.args.get('ft', type=str)
    sm = request.args.get('sm', type=str)

    if sm.lower() == 'true' or session['username'] not in ['admin', 'social']:
        google = False
        print('Google NLP not in use')
    else:
        google = True
        print('Google NLP in use')

    filter_words = [x.strip() for x in ft.split(',')]

    p, c = fn.GetFacebookComments(post_id=kw, graphAPI=graph, limit=mt)

    # process information of the post
    try:
        del p['id']
        my_table_p = pd.DataFrame(list(p.values())).transpose()
        try:
            my_table_p['post'] = my_table_p[0]
            my_table_p['time'] = pd.to_datetime(my_table_p[1]).dt.strftime('%Y/%m/%d %H:%M')
            my_table_p = my_table_p[['post', 'time']]
        except ValueError:
            my_table_p['post'] = my_table_p[1]
            my_table_p['time'] = pd.to_datetime(my_table_p[0]).dt.strftime('%Y/%m/%d %H:%M')
            my_table_p = my_table_p[['post', 'time']]
    except:
        my_table_p = pd.DataFrame([" ".join(list(p.values())), "Unknown"]).transpose()
        my_table_p.columns = ['post', 'time']

    # analyse comments
    results, msgs = fn.AnalyseFacebookComments(comments=c, filter_by=filter_words, Google=google)
    msgs = ' '.join(msgs)

    data = pd.DataFrame(results).transpose()

    data.columns = ['txt', 'emoji_scores', 'emoji_extracted', 'google_scores',
                    'google_magnitude', 'textblob_scores', 'vader_scores', 'textacy_scores',
                    'created_at']

    if google:
        data['scores'] = ((data.google_scores * data.google_magnitude
                           + data.textblob_scores + data.vader_scores)
                          / 3).astype(np.double).round(2)
    else:
        data['scores'] = ((data.textblob_scores + data.vader_scores)
                          / 2).astype(np.double).round(2)

    avg_scores = data['scores'].mean()
    msgs = msgs + ' The average sentiment score is {0:.2f}'.format(avg_scores)

    my_table = data[['txt', 'created_at', 'scores']]
    my_table['created_at'] = pd.to_datetime(my_table.created_at).dt.strftime('%Y/%m/%d %H:%M')

    df = data[['txt', 'scores', 'created_at']]
    df['key'] = pd.to_datetime(df.created_at).dt.strftime('%Y/%m/%d %H:%M')

    raw_text_list = df.txt.tolist()
    label = []
    for item in raw_text_list:
        label.append("\n".join(textwrap.wrap(item, width=30)))

    df['label'] = label
    df = df[['key', 'scores', 'label']]

    return jsonify(data=df.values.tolist(),
                   msgs=msgs,
                   my_table=json.loads(my_table.to_json(orient="split"))["data"],
                   my_table_p=json.loads(my_table_p.to_json(orient="split"))["data"])


# OzBargain analysis results
@app.route('/_get_graph_oz')
def get_graph_oz():

    kw = request.args.get('kw', type=str)
    mt = request.args.get('mt', type=int)
    mc = request.args.get('mc', type=int)
    sm = request.args.get('sm', type=str)
    ac = request.args.get('ac', type=str)

    if sm.lower() == 'true' or session['username'] not in ['admin', 'social']:
        google = False
        print('Google NLP not in use')
    else:
        google = True
        print('Google NLP in use')

    ex_inactive = (ac.lower() == 'true')

    nodes = fn.SearchOzbargain(searchQuery=kw, excludeInvalid=ex_inactive, maxNodeNum=mt)

    if nodes == 'Connection refused.':
        return jsonify(data=[],
                       msgs=nodes,
                       my_table=[],
                       my_table_p=[])

    comments = []
    for node in nodes:
        comments.append(fn.GetComments(node=node, maxNumComments=mc))

    my_table_p = pd.DataFrame(comments).drop(columns=['comments'])

    results, msgs = fn.AnalyseOzbargainComments(comments=comments, Google=google)
    msgs = ' '.join(msgs)

    data = pd.DataFrame(results).transpose()

    data.columns = ['txt', 'emoji_scores', 'emoji_extracted', 'google_scores',
                    'google_magnitude', 'textblob_scores', 'vader_scores', 'textacy_scores',
                    'created_at', 'vote']

    if google:
        data['scores'] = ((data.google_scores * data.google_magnitude
                           + data.textblob_scores + data.vader_scores)
                          / 3).astype(np.double).round(2)
    else:
        data['scores'] = ((data.textblob_scores + data.vader_scores)
                          / 2).astype(np.double).round(2)

    avg_scores = data['scores'].mean()
    msgs = msgs + ' The average sentiment score is {0:.2f}'.format(avg_scores)

    my_table = data[['txt', 'created_at', 'scores', 'vote']]
    my_table['created_at'] = pd.to_datetime(my_table.created_at).dt.strftime('%Y/%m/%d %H:%M')

    df = data[['txt', 'scores', 'created_at']]
    df['key'] = pd.to_datetime(df.created_at).dt.strftime('%Y/%m/%d %H:%M')

    raw_text_list = df.txt.tolist()
    label = []
    for item in raw_text_list:
        label.append("\n".join(textwrap.wrap(item, width=30)))

    df['label'] = label
    df = df[['key', 'scores', 'label']]

    return jsonify(data=df.values.tolist(),
                   msgs=msgs,
                   my_table=json.loads(my_table.to_json(orient="split"))["data"],
                   my_table_p=json.loads(my_table_p.to_json(orient="split"))["data"])


if __name__ == '__main__':
    app.debug = True
    app.run()

