# visualization of the SuicideWatch Posts in Post 16 - Post 25
import pandas as pd
import argparse

if __name__ == "__main__":
	df = pd.read_csv('post_16_25.csv')
	suicideWatch_df = df.loc[df['subreddit'] == 'SuicideWatch']
	suicideWatch_df.to_csv('post_16_25_SuicideWatch.csv')