from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import datetime

import credentials


class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def GetTweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(API.search,
                            q="Giraffes",
                            since="2015-10-10",
                            until="2015-10-11").items():
            tweets.append(tweet)
        return tweets


class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(credentials.consumer_key,
                            credentials.consumer_secret)
        auth.set_access_token(credentials.access_token,
                              credentials.access_token_secret)
        return auth


class TwitterListener(StreamListener):

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            # Return false on data method in case rate limited
            return False
        print(status)


class TweetStreamer():
    """
    Class for streaming and processing tweets
    """

    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):

        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        stream.filter(track=hash_tag_list)


if __name__ == "__main__":
    hash_tag_list = ['Love Island']
    fetched_tweets_filename = "tweets.json"

    twitter_client = TwitterClient('pycon')
    print(twitter_client)

    # twitter_streamer = TweetStreamer()
    # twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
