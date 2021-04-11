import sys
import csv
import tweepy

# Get these values from your application settings.
CONSUMER_KEY = ''
CONSUMER_SECRET = ''

# Get these values from the "My Access Token".
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''


def get_tweets(api, search, items=100000):
    """Make request to twitter api to obtain tweets for given search query in english"""
    cursor = tweepy.Cursor(api.search, q=search, lang='en', tweet_mode='extended', exclude_replies=True).items(items)
    tweets = [parse_status(status) for status in cursor]
    return tweets


def parse_status(status):
    """Parse useful information from status"""
    return status.full_text, status.retweet_count, status.favorite_count


def main():
    search = sys.argv[1]
    items = int(sys.argv[2])

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    tweets = set(get_tweets(api, search, items))  # To avoid duplicates
    with open('../data/tweets_{}.csv'.format(search), 'a') as f:
        write = csv.writer(f)
        write.writerow(['Full_text', 'Retweet_count', 'Favorite_count'])  # Write column names
        write.writerows(tweets)


if __name__ == "__main__":
    main()
