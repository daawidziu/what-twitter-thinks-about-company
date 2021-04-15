# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import random
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd

df = pd.read_csv('tweets_nlp.csv')
words = json.load(open('words_top.json'))

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])


def draw_tweets(dataframe):
    """Card with count of tweets"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('TOTAL TWEETS', className="card-title"),
            html.H3('{:,}'.format(len(dataframe)).replace(',', ' '), className="card-text")
        ])
    ])


def draw_retweets(dataframe):
    """Card with total retweets count"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('TOTAL RETWEETS', className="card-title"),
            html.H3('{:,}'.format(dataframe['Retweet_count'].sum()).replace(',', ' '), className="card-text")
        ])
    ])


def draw_favourite(dataframe):
    """Card with total favourites count"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('TOTAL FAVOURITES', className="card-title"),
            html.H3('{:,}'.format(dataframe['Favorite_count'].sum()).replace(',', ' '), className="card-text")
        ])
    ])


def draw_random(dataframe):
    """Card with one sample tweet (to be used in draw_sample)"""
    index = random.randint(0, len(dataframe))
    tweet = dataframe.iloc[index, 1]
    return dbc.Card([
        dbc.CardHeader('Somebody says:'),
        dbc.CardBody(tweet)
    ])


def draw_samples(dataframe):
    """Card with 4 sample tweets"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('Sample tweets'),
            html.Br(),
            draw_random(dataframe),
            html.Br(),
            draw_random(dataframe),
            html.Br(),
            draw_random(dataframe),
            html.Br(),
            draw_random(dataframe),
        ])
    ])


def fig_sentiment(dataframe):
    """Figure of sentiment intensity polarity"""
    neu_num = sum(dataframe['Compound'] == 0)
    pos_num = sum(dataframe['Compound'] > 0)
    neg_num = sum(dataframe['Compound'] < 0)
    labels = ['Positive', 'Neutral', 'Negative']
    fig = go.Figure(data=go.Pie(labels=labels, values=[pos_num, neu_num, neg_num], hole=0.55))
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                      marker=dict(colors=['ForestGreen', 'Gray', 'FireBrick']))
    fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    return fig


def draw_sentiment(dataframe):
    """Card with graph of sentiment polarity"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('Sentiment Polarity'),
            dcc.Graph(figure=fig_sentiment(dataframe))
        ])
    ])


def fig_tsne(dataframe):
    """Figure of topics projection on to 2D space using t-SNE"""
    fig = go.Figure(data=go.Scattergl(x=dataframe['X-tsne'], y=dataframe['Y-tsne'],
                                      mode='markers', marker_color=dataframe['Topic'],
                                      text=dataframe['Full_text'], hoverinfo=['text']))
    fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    return fig


def draw_tsne(dataframe):
    """Card with graph of topics projection"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('Projection of topic distribution'),
            dcc.Graph(figure=fig_tsne(dataframe))
        ])
    ])


@app.callback(Output('top_words', 'figure'),
              Input('btn-0', 'n_clicks'),
              Input('btn-1', 'n_clicks'),
              Input('btn-2', 'n_clicks'),
              Input('btn-3', 'n_clicks'),
              Input('btn-4', 'n_clicks'), )
def fig_words(btn_0, btn_1, btn_2, btn_3, btn_4):
    """Figure of most relevant words for each topic,
    Callback connecting """
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0][-1]
    top_words = words[button_id]
    fig = go.Figure(data=go.Bar(x=list(top_words.values()), y=list(top_words.keys()),
                                orientation='h', marker={'color': 'ForestGreen'}))
    fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    return fig


def draw_words(dataframe):
    """Card with most relevant words for each topic"""
    return dbc.Card([
        dbc.CardBody([
            html.H4('Most important words for each topic'),
        ]),
        dbc.ButtonGroup([
            dbc.Button('Topic 1', id='btn-0', outline=True, color="success"),
            dbc.Button('Topic 2', id='btn-1', outline=True, color="success"),
            dbc.Button('Topic 3', id='btn-2', outline=True, color="success"),
            dbc.Button('Topic 4', id='btn-3', outline=True, color="success"),
            dbc.Button('Topic 5', id='btn-4', outline=True, color="success")
        ]),
        dbc.CardBody([
            dcc.Graph(id='top_words')
        ])
    ])


app.layout = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.H1('WHAT TWITTER THINK ABOUT NOKIA'),
            html.Br(),
            dbc.Row([
                dbc.Col(draw_tweets(df), width=4),
                dbc.Col(draw_retweets(df), width=4),
                dbc.Col(draw_favourite(df), width=4)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(draw_tsne(df), width=8),
                dbc.Col(draw_sentiment(df), width=4)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(draw_samples(df), width=8),
                dbc.Col(draw_words(df), width=4),
            ])
        ])
    ], color='dark')
])

if __name__ == "__main__":
    app.run_server()
