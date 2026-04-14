#!/usr/bin/env python3
"""
Script to fetch top 5 trending Twitter posts and summarize them using Ollama.

Requirements:
    pip install tweepy requests

Twitter API Setup:
    1. Go to https://developer.twitter.com/
    2. Create a project and get API keys
    3. Add the credentials below or set as environment variables
"""

import tweepy
import sys
import os
import json
from prompt_ollama import prompt_ollama_chat

# Load configuration from config.json
def load_config():
    """Load configuration from config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: config.json is not valid JSON", file=sys.stderr)
        sys.exit(1)

config = load_config()
TWITTER_API_KEY = config["twitter"]["api_key"]
TWITTER_API_SECRET = config["twitter"]["api_secret"]
TWITTER_ACCESS_TOKEN = config["twitter"]["access_token"]
TWITTER_ACCESS_TOKEN_SECRET = config["twitter"]["access_token_secret"]
TWITTER_BEARER_TOKEN = config["twitter"]["bearer_token"]


def get_trending_tweets(count: int = 5) -> list[dict]:
    """
    Fetch top trending tweets.
    
    Args:
        count: Number of tweets to fetch (default: 5)
    
    Returns:
        List of tweet dictionaries with 'author' and 'text' keys
    """
    try:
        # Authenticate with Twitter API v2
        client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )
        
        # Search for trending tweets
        # Using a broader search query to find trending content
        query = "-is:retweet lang:en"
        tweets = client.search_recent_tweets(
            query=query,
            max_results=count,
            tweet_fields=["author_id", "created_at", "public_metrics"],
            expansions=["author_id"],
            user_fields=["username"]
        )
        
        if not tweets.data:
            print("No tweets found. Check your API credentials and rate limits.", file=sys.stderr)
            return []
        
        # Extract tweet data
        users = {user.id: user.username for user in tweets.includes["users"]}
        
        tweet_list = []
        for tweet in tweets.data:
            tweet_list.append({
                "author": users.get(tweet.author_id, "Unknown"),
                "text": tweet.text,
                "created_at": tweet.created_at
            })
        
        return tweet_list
    
    except tweepy.TweepyException as e:
        print(f"Twitter API Error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error fetching tweets: {e}", file=sys.stderr)
        return []


def summarize_tweets(tweets: list[dict]) -> str:
    """
    Use Ollama to summarize the trending tweets.
    
    Args:
        tweets: List of tweet dictionaries
    
    Returns:
        Summary from the AI model
    """
    if not tweets:
        return "No tweets to summarize."
    
    # Format tweets for the prompt
    tweets_text = "\n".join([
        f"@{tweet['author']}: {tweet['text']}"
        for tweet in tweets
    ])
    
    # Create messages for chat API
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes trending social media posts. Provide concise, informative summaries highlighting the key topics and sentiment."
        },
        {
            "role": "user",
            "content": f"Please summarize these top trending Twitter posts:\n\n{tweets_text}\n\nProvide a brief overview of the main topics and trends."
        }
    ]
    
    # Get summary from Ollama
    print("\nGenerating summary...\n")
    summary = prompt_ollama_chat(messages, stream=True)
    
    return summary


def main():
    """Main function to fetch and summarize trending tweets."""
    
    # Validate API credentials
    if TWITTER_API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Twitter API credentials not configured.", file=sys.stderr)
        print("\nTo set up Twitter API credentials:", file=sys.stderr)
        print("  1. Go to https://developer.twitter.com/", file=sys.stderr)
        print("  2. Create a project and get your API keys", file=sys.stderr)
        print("  3. Update config.json with your credentials:", file=sys.stderr)
        print("     - twitter.api_key", file=sys.stderr)
        print("     - twitter.api_secret", file=sys.stderr)
        print("     - twitter.access_token", file=sys.stderr)
        print("     - twitter.access_token_secret", file=sys.stderr)
        print("     - twitter.bearer_token", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("Fetching top 5 trending Twitter posts...")
    print("=" * 60)
    
    # Fetch tweets
    tweets = get_trending_tweets(count=5)
    
    if not tweets:
        print("Failed to fetch tweets.", file=sys.stderr)
        sys.exit(1)
    
    # Display fetched tweets
    print(f"\nFound {len(tweets)} tweets:\n")
    for i, tweet in enumerate(tweets, 1):
        print(f"{i}. @{tweet['author']}")
        print(f"   {tweet['text'][:100]}...")
        print()
    
    # Summarize with Ollama
    print("=" * 60)
    print("AI Summary:")
    print("=" * 60)
    summary = summarize_tweets(tweets)
    
    print("\n" + "=" * 60)
    print("Summary complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
