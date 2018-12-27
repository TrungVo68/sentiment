def TwitterAPILogin(AppOnly = True):
    import tweepy
    
    consumer_key = 'WuV9EpIRa3eyzoq3oG8LV6le3'
    consumer_secret = 'getqMOj0pTQc2OMfHq9ck0ybGtaPDw04oELEBNkXOukboBfoOz'
    access_token = '1020160365961756672-icm2oRhQycPzVNknRTOBdX6Oq7olWK'
    access_token_secret = 'PxpNXMBL0mT33w77kfgwmKQwFF3ROBhFFugzzdYbbXJLW'
    
    if AppOnly:
        auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    else:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        
    return api


def FacebookAPILogin(key = 'EAAGT0Xx9qWgBAI3L6kmOIj4ZA5RHHOsXZCNPXduJPUvxQyNF71u80GG0CkBAFoB9jnfxrcGrnbshN6UZBaJZA28zpa5sSZAtct82D25Gf4U4rmwcNAkG4AfxRYlZCuKTAh6oybZAECY3EqKfsg0VN1x2AmIRv5Sek6j22xVbT47bjUFQUE55e6q',
                     API_version = '2.7'):
    import facebook
    graph = facebook.GraphAPI(access_token = key, version = API_version)
    return graph


def GetTweets(api = None, searchQuery = None, maxTweets = 1000, tweetsPerQry = 100, sinceId = None, max_id = -1, lang = "en"):
    if not api:
        print("API must be specified. No default value")
        return
    if not searchQuery:
        print("Please specify search words.")
        return
    
    import tweepy
    
    if maxTweets < tweetsPerQry:
        tweetsPerQry = maxTweets
    
    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    
    tweetCount = 0
    tweetExtracted = []
    print("Downloading max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang=lang, tweet_mode='extended')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang=lang, tweet_mode='extended',
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang=lang, tweet_mode='extended',
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang=lang, tweet_mode='extended',
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            
            for tweet in new_tweets:
                tweetExtracted.append(tweet._json)
            
            tweetCount += len(new_tweets)
                        
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("Error : " + str(e))
            break
    
    print ("Downloaded {0} tweets in total.".format(tweetCount))
    return tweetExtracted



def GetFacebookComments(post_id = None, graphAPI = None, limit = 2000, fileName = None):
    if not graphAPI:
        print("ERROR: graph API must be specified. No default value")
        return
    if not post_id:
        print("Please specify post ID.")
        return
    
    # import jsonpickle
    
    p = graphAPI.get_object(post_id)
    c = graphAPI.get_connections(post_id, 'comments', limit = limit)
    comments = c['data']
    
    # write extracted content to file
    """
    if fileName is not None:
        with open(fileName, 'w') as f:               
            for post in comments:
                f.write(jsonpickle.encode(post, unpicklable=False) + '\n')
    """
    return p, comments


def text_prep(text, sentiment_tokenizer):
    import pandas as pd
    from keras.preprocessing.sequence import pad_sequences
    text = pd.Series(str(text))
    max_sequence_length = 32
    out = pad_sequences(sentiment_tokenizer.texts_to_sequences(text), maxlen=max_sequence_length)
    return out


def my_sentiment(tweet, sentiment_model, sentiment_tokenizer):
    # import os
    from bs4 import BeautifulSoup
    import re
    # from keras.models import load_model
    # import pickle
    from keras import backend as K

    # data cleaning
    tweet = BeautifulSoup(tweet, 'lxml').get_text()
    tweet = re.sub('RT ', '', tweet)  # remove the RT label
    tweet = re.sub('https?://[A-Za-z0-9./]+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # be careful about this as it may remove the subject/object.
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet)  # be careful about not to collapse white spaces between words

    path_model = "G:\\Trung\\Sentiment\\model\\"
    # sentiment_model = load_model(path_model + 'sentiment_2channels_bidirectionalLSTM1.h5')
    # sentiment_tokenizer = pickle.load(open(path_model + 'sentiment_tokenizer.sav', 'rb'))

    ww_scores = []
    my_text = text_prep(tweet, sentiment_tokenizer)
    # sentiment_model._make_predict_function()
    K.clear_session()
    ww_scores.append(sentiment_model.predict(my_text)[0][1])

    return ww_scores


def AnalyseTweets(tweets = None, keep_RT = False, filter_by = None, Google = True,
                  time_format_in = '%a %b %d %H:%M:%S %z %Y', time_diff = 10):
    
    if not tweets:
        print("ERROR: tweets must be specified. No default value.")
        return
    
    import tweepy
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    # from dateutil import tz
    
    from bs4 import BeautifulSoup
    import re

    import os    
    
    # import emoji engine (self-defined)
    from emoji_engine import Engine
    emoji_engine = Engine()
    emoji_extracted = []
    emoji_scores = []
    
    # Google
    if Google:
        # login to Google language API
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']="wx-bq-poc-service-key.json"
        from google.cloud import language
        client = language.LanguageServiceClient()
        
        from google.cloud.language import enums
        from google.cloud.language import types
    
    google_scores = []
    google_magnitude = []
    
    # textblob
    from textblob import TextBlob
    textblob_scores = []
    
    # vader
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = []

    # LOAD HOME-MADE MODEL HERE
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences
    import pickle

    ww_scores = []
    sentiment_model = load_model(os.getcwd() + '\\model\\sentiment_2channels_bidirectionalLSTM3.h5')
    sentiment_tokenizer = pickle.load(open(os.getcwd() + '\\model\\sentiment_tokenizer.sav', 'rb'))

    # # PRE-DEFINE TEXT CONSTANT
    # def text_prep(text):
    #     text = pd.Series(text)
    #     max_sequence_length = 32
    #     return pad_sequences(sentiment_tokenizer.texts_to_sequences(text), maxlen=max_sequence_length)


    # textacy
    textacy_scores = []
    
    # initialization for storing results
    txt = []
    numbers_list = []
    number = 0
    number_RT = 0
    number_irrelevant = 0
    
    created_at = []
    hashtags = []
    user_mentions = []
    in_reply_to_status_id = []    
    favorite_count = []
    retweet_count = []
    place = []
    
    location = []
    favourites_count = []
    followers_count = []
    friends_count = []
    listed_count = []
    statuses_count = []
    
    if filter_by is not None:
        if isinstance(filter_by, str):
            filter_by = [filter_by]
        filter_by = [x.lower() for x in filter_by]
    
    # loop through all tweets
    ids_seen = [] # placeholder for ids that have been processed. Used to remove duplicated tweets
    number_dup = 0
    
    for t in tweets:
        try:
            if t['id'] in ids_seen:
                number_dup += 1
                continue
            ids_seen.append(t['id'])
            
            tweet = t['full_text']
            
            if filter_by is not None:
                if not any(x in tweet.lower() for x in filter_by):
                    number_irrelevant += 1
                    continue
            
            if tweet[:3] == 'RT ':
                number_RT += 1
                if not keep_RT:
                    continue
            
            txt.append(tweet)
            
            emoji_extract_result = emoji_engine.extract_emoji_info(tweet)
            
            if len(emoji_extract_result) > 0:
                df = pd.DataFrame(emoji_extract_result).drop_duplicates()
                emoji_extracted.append(df[0].tolist())
                emoji_scores.append(np.sum(df[1]))
            else:
                emoji_extracted.append([])
                emoji_scores.append(0)
            
            # data cleaning
            tweet = BeautifulSoup(tweet, 'lxml').get_text()
            tweet = re.sub('RT ', '', tweet) # remove the RT label
            tweet = re.sub('https?://[A-Za-z0-9./]+', '', tweet)
            tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet) # be careful about this as it may remove the subject/object.
            tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet) # be careful about not to collapse white spaces between words
            
            # Google
            if Google:
                document = types.Document(content = tweet, type = enums.Document.Type.PLAIN_TEXT)
                sentiment = client.analyze_sentiment(document = document).document_sentiment
                google_scores.append(sentiment.score)
                google_magnitude.append(sentiment.magnitude)
            
            
            # textblob
            polarity = TextBlob(tweet).sentiment.polarity
            textblob_scores.append(polarity)
            
            # vader
            vader_scores.append(vader_analyzer.polarity_scores(tweet)['compound'])

            # HOME-MADE MODEL HERE
            out = sentiment_model.predict(text_prep(tweet))
            ww_scores.append(2*out[0][1]-1)


            # other useful information about the tweet
            # other useful information about the comment
            local_time = datetime.strptime(t['created_at'], time_format_in) + timedelta(hours = time_diff) # convert UTC to AEST
            created_at.append(local_time)

            hashtags.append([x['text'] for x in t['entities']['hashtags']])
            user_mentions.append([x['screen_name'] for x in t['entities']['user_mentions']])
            in_reply_to_status_id.append(t['in_reply_to_status_id']) # if NULL, then this is original
            favorite_count.append(t['favorite_count']) # approximately how many times this Tweet has been liked by Twitter users
            retweet_count.append(t['retweet_count']) # number of times this Tweet has been retweeted
            if not t['place']:
                place.append([])
            else:
                place.append(t['place']['name'])
            
            # other userful information about the account
            location.append(t['user']['location'])
            favourites_count.append(t['user']['favourites_count']) # number of Tweets this user has liked in the account’s lifetime
            followers_count.append(t['user']['followers_count']) # number of followers this account currently has
            friends_count.append(t['user']['friends_count']) # number of users this account is following (AKA their “followings”)
            listed_count.append(t['user']['listed_count']) # number of public lists that this user is a member of
            statuses_count.append(t['user']['statuses_count']) # number of Tweets (including retweets) issued by the user
            
            number += 1

            if number % 100 == 0:
                print("Completed analysing {} tweets...".format(number))
            numbers_list.append(number)

        except tweepy.TweepError as e:
            print(e.reason)

        except StopIteration:
            break
    
    if keep_RT:
        msg1 = "{} tweets and {} retweets have been analysed.".format(number - number_RT, number_RT)
        msg2 = "{} duplicates were removed.".format(number_dup)
        print(msg1 + "\n" + msg2)
        msgs = [msg1, msg2]
    else:
        msg1 = "{} tweets have been analysed.".format(number)
        msg2 = "{} retweets were discarded.".format(number_RT)
        if number_dup > 0:
            msg3 = "{} duplicates were removed.".format(number_dup)
        else:
            msg3 = ""
        print(msg1 + "\n" + msg2 + "\n" + msg3)
        msgs = [msg1, msg2, msg3]
    
    if number_irrelevant > 0:
        msg4 = "{} irrelevant tweets were discarded. ".format(number_irrelevant) 
        print(msg4)
        msgs.append(msg4)
    
    out = [txt, emoji_scores, emoji_extracted, google_scores, google_magnitude, textblob_scores, vader_scores, ww_scores, textacy_scores,
           created_at, hashtags, user_mentions, in_reply_to_status_id, favorite_count, retweet_count, place,
           location, favourites_count, followers_count, friends_count, listed_count, statuses_count, numbers_list]

    return out, msgs



def AnalyseFacebookComments(comments = None, filter_by = None, Google = True, 
                            time_format_in = '%Y-%m-%dT%H:%M:%S%z', time_diff = 10):
    
    if not comments:
        print("Please specify comments to be analysed.")
        return
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    from bs4 import BeautifulSoup
    import re
    import os    
            
    # import emoji engine (self-defined)
    from emoji_engine import Engine
    emoji_engine = Engine()
    emoji_extracted = []
    emoji_scores = []
    
    # login to Google language API
    if Google:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']="wx-bq-poc-service-key.json"
        from google.cloud import language
        client = language.LanguageServiceClient()
        
        from google.cloud.language import enums
        from google.cloud.language import types
    
    google_scores = []
    google_magnitude = []
    
    # textblob
    from textblob import TextBlob
    textblob_scores = []
    
    # vader
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = []
    
    # textacy
    textacy_scores = []
    
    # initialization for storing results
    txt = []
    numbers_list = []
    number = 0
    number_irrelevant = 0    
    
    created_at = []
       
    # loop through all comments    
    for c in comments:
        try:
            comment = c['message']
            
            if filter_by is not None:
                if isinstance(filter_by, str):
                    filter_by = [filter_by]
                
                if not any(x.lower() in comment.lower() for x in filter_by):
                    number_irrelevant += 1
                    continue
            
            txt.append(comment)

            emoji_extract_result = emoji_engine.extract_emoji_info(comment)
            
            if len(emoji_extract_result) > 0:
                df = pd.DataFrame(emoji_extract_result).drop_duplicates()
                emoji_extracted.append(df[0].tolist())
                emoji_scores.append(np.sum(df[1]))
            else:
                emoji_extracted.append([])
                emoji_scores.append(0)
            
            # data cleaning
            comment = BeautifulSoup(comment, 'lxml').get_text()
            comment = re.sub('RT ', '', comment) # remove the RT label
            comment = re.sub('https?://[A-Za-z0-9./]+', '', comment)
            comment = re.sub(r'@[A-Za-z0-9]+', '', comment) # be careful about this as it may remove the subject/object.
            comment = re.sub('[^A-Za-z0-9]+', ' ', comment) # be careful about not to collapse white spaces between words
            
            # Google
            if Google:
                document = types.Document(content = comment, type = enums.Document.Type.PLAIN_TEXT)
                sentiment = client.analyze_sentiment(document = document).document_sentiment
                google_scores.append(sentiment.score)
                google_magnitude.append(sentiment.magnitude)

            
            # textblob
            polarity = TextBlob(comment).sentiment.polarity
            textblob_scores.append(polarity)
            
            # vader
            vader_scores.append(vader_analyzer.polarity_scores(comment)['compound'])
            
            
            # other useful information about the comment
            local_time = datetime.strptime(c['created_time'], time_format_in) + timedelta(hours = time_diff) # convert UTC to AEST
            created_at.append(local_time)

            number += 1
            
            if number % 100 == 0:
                print("Completed analysing {} Facebook comments...".format(number))
            numbers_list.append(number)
            
        except StopIteration:
            break
    
    if number_irrelevant == 0:
        msg1 = "{} Facebook comments have been analysed.".format(number) 
        print(msg1)
        msgs = [msg1]
    else:
        msg1 = "{} Facebook comments have been analysed.".format(number)
        msg2 = "{} irrelevant comments were discarded.".format(number_irrelevant) 
        print(msg1 + "\n" + msg2)
        msgs = [msg1, msg2]
        
    out = [txt, emoji_scores, emoji_extracted, google_scores, google_magnitude, 
           textblob_scores, vader_scores, textacy_scores, created_at]
    
    return out, msgs



def SearchOzbargain(searchQuery = None, excludeInvalid = True, maxNodeNum = 10):
    # search on OzBargain and return node references for available offers
    
    if not searchQuery:
        print("ERROR: Please specify a keyword/phrase to search.")
        return
    
    import requests
    from bs4 import BeautifulSoup

    page_url = 'https://www.ozbargain.com.au'
    try:
        page = requests.get(page_url + '/search/node/' + searchQuery).text
    except requests.exceptions.ConnectionError:
        msg = "Connection refused."
        return msg
    
    offer_nodes = []
    
    next_page = True
    node_num = 0
    
    while node_num < maxNodeNum and next_page:
        soup = BeautifulSoup(page, 'lxml')
        
        # offers related to store
        for item in soup.find_all('div', {'class': 'title'}):
            if node_num >= maxNodeNum:
                continue

            if excludeInvalid:
                offer = requests.get(page_url + item.a.attrs['href']).text
                offer_soup = BeautifulSoup(offer, 'lxml')
                if offer_soup.find('span', {'class': 'tagger expired'}) is not None:
                    continue

            node = item.a.attrs['href']
            
            # exclude non-offer links
            if 'node/' not in node:
                continue
            
            offer_nodes.append(node)
            node_num += 1
    
        # offers shown on page    
        for item in soup.find_all('dt', {'class': 'title'}):
            if node_num >= maxNodeNum:
                continue
            
            if excludeInvalid:
                if item.find('span', {'class': 'tagger expired'}) is not None:
                    continue
            
            node = item.a.attrs['href']
            
            # exclude non-offer links
            if 'node/' not in node:
                continue
                        
            offer_nodes.append(node)
            node_num += 1
            
        # determine whether there is a next page
        pager = soup.find('ul', {'class': 'pager'})
        
        if pager is None:
            next_page = False
        elif pager.find('a', {'class': 'pager-next'}) is None:
            next_page = False
        else:
            next_url = page_url + pager.find('a', {'class': 'pager-next'}).attrs['href']
            page = requests.get(next_url).text
    
    # just in case
    offer_nodes = list(set(offer_nodes))

    return offer_nodes


def GetComments(node = None, maxNumComments = 20):
    # get comments on a page
    
    if node is None:
        print('ERROR: Please specify a node.')
    
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime, timedelta

    page_url = 'https://www.ozbargain.com.au'
    
    url = page_url + node
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'lxml')

    # get title
    if soup.find('meta', {'property': 'og:title'}) is not None:
        title = soup.find('meta', {'property': 'og:title'}).attrs['content']
    else:
        title = ""
    
    # get status and vote up and down
    if len(soup.find_all('span', {'class': 'nvb'})) != 2:
        status = 'topic'
        vote_up = 0
        vote_dn = 0
    else:
        up, dn = soup.find_all('span', {'class': 'nvb'})
        vote_up = int(up.text)
        vote_dn = int(dn.text)
        
        if soup.find('span', {'class': 'tagger expired'}) is None:
            status = 'active'
        else:
            status = soup.find('span', {'class': 'tagger expired'}).text
    
    # go through pages to get all comments
    comments = []
    next_page = True
    comment_num = 0
    
    while comment_num < maxNumComments and next_page:
        soup = BeautifulSoup(page, 'lxml')
        
        # offers related to store
        for item in soup.find_all('div', {'class': 'comment'}):
            if comment_num >= maxNumComments:
                continue

            # text
            text = ''
            for t in item.find_all('p'):
                text = text + t.text
            text = text.replace('\n', ' ')
            
            # vote
            voting = item.find('span', {'class': 'cvc'})
            if voting is None:
                vote = 0
            else:
                vote = voting.text.replace('votes', '').replace('vote', '').strip()
                if vote == '':
                    vote = 0
                else:
                    vote = int(vote)
                
            # time
            ts = int(item.attrs['data-ts'])
            time = datetime.utcfromtimestamp(ts) + timedelta(hours = 10)
            
            content = {'text': text,
                       'vote': vote,
                       'time': time}
            
            comments.append(content)
            comment_num += 1
    
        # determine whether there is a next page
        pager = soup.find('ul', {'class': 'pager'})
        
        if pager is None:
            next_page = False
        elif pager.find('a', {'class': 'pager-next'}) is None:
            next_page = False
        else:
            next_url = page_url + pager.find('a', {'class': 'pager-next'}).attrs['href']
            page = requests.get(next_url).text
    
    extracted = {'title': title,
                 'status': status,
                 'vote_up': vote_up,
                 'vote_dn': vote_dn,
                 'comments': comments}
    
    return extracted



def AnalyseOzbargainComments(comments = None, Google = True):
    
    if not comments:
        print("Please specify comments to be analysed.")
        return
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    from bs4 import BeautifulSoup
    import re
    import os    

    # import emoji engine (self-defined)
    from emoji_engine import Engine
    emoji_engine = Engine()
    emoji_extracted = []
    emoji_scores = []
    
    # login to Google language API
    if Google:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']="wx-bq-poc-service-key.json"
        from google.cloud import language
        client = language.LanguageServiceClient()
        
        from google.cloud.language import enums
        from google.cloud.language import types
    
    google_scores = []
    google_magnitude = []
    
    # textblob
    from textblob import TextBlob
    textblob_scores = []
    
    # vader
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = []
    
    # textacy
    textacy_scores = []
    
    # initialization for storing results
    txt = []
    numbers_list = []
    vote = []
    number = 0
    
    created_at = []
       
    # loop through all comments    
    for c in comments:
        try:
            for cc in c['comments']:
                comment = cc['text']
                txt.append(comment)
                emoji_extract_result = emoji_engine.extract_emoji_info(comment)
            
                if len(emoji_extract_result) > 0:
                    df = pd.DataFrame(emoji_extract_result).drop_duplicates()
                    emoji_extracted.append(df[0].tolist())
                    emoji_scores.append(np.sum(df[1]))
                else:
                    emoji_extracted.append([])
                    emoji_scores.append(0)
                
                # data cleaning
                comment = BeautifulSoup(comment, 'lxml').get_text()
                comment = re.sub('RT ', '', comment) # remove the RT label
                comment = re.sub('https?://[A-Za-z0-9./]+', '', comment)
                comment = re.sub(r'@[A-Za-z0-9]+', '', comment) # be careful about this as it may remove the subject/object.
                comment = re.sub('[^A-Za-z0-9]+', ' ', comment) # be careful about not to collapse white spaces between words
                
                # Google
                if Google:
                    document = types.Document(content = comment, type = enums.Document.Type.PLAIN_TEXT)
                    sentiment = client.analyze_sentiment(document = document).document_sentiment
                    google_scores.append(sentiment.score)
                    google_magnitude.append(sentiment.magnitude)
    
    
                
                #************************************#
                # add entity analysis and syntax analysis
                # ..........
                
                
                # textblob
                polarity = TextBlob(comment).sentiment.polarity
                textblob_scores.append(polarity)
                
                # vader
                vader_scores.append(vader_analyzer.polarity_scores(comment)['compound'])
                
    
                # other useful information about the comment
                created_at.append(cc['time'])
                vote.append(cc['vote'])
    
                number += 1
                
                if number % 100 == 0:
                    print("Completed analysing {} OzBargain comments...".format(number))
                numbers_list.append(number)
            
        except StopIteration:
            break
    
    msg1 = "{} offers were extracted.".format(len(comments))
    msg2 = "{} comments were analysed.".format(number) 
    print(msg1 + "\n" + msg2)
    msgs = [msg1, msg2]
        
    out = [txt, emoji_scores, emoji_extracted, google_scores, google_magnitude, 
           textblob_scores, vader_scores, textacy_scores, created_at, vote]
    
    return out, msgs


