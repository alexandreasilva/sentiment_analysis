import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from collections import Counter
from wordcloud import WordCloud
import datetime
import argparse

# Download necessary NLTK resources
download('vader_lexicon')
download('punkt')
download('stopwords')

class SentimentAnalyzer:
    """Main class for sentiment analysis of text data."""
    
    def __init__(self):
        """Initialize the sentiment analyzer with VADER."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean text by removing special characters, links, etc."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions (for social media data)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER."""
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return {
                'compound': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'sentiment': 'neutral'
            }
            
        scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
        
        # Determine overall sentiment based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        scores['sentiment'] = sentiment
        return scores
    
    def analyze_batch(self, texts):
        """Analyze sentiment for a batch of texts."""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def extract_keywords(self, texts, sentiment_filter=None, top_n=20):
        """Extract most common keywords from texts, optionally filtered by sentiment."""
        all_words = []
        
        for i, text in enumerate(texts):
            if sentiment_filter is not None and self.analyze_sentiment(text)['sentiment'] != sentiment_filter:
                continue
                
            cleaned_text = self.clean_text(text)
            tokens = word_tokenize(cleaned_text)
            
            # Remove stop words
            filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            all_words.extend(filtered_tokens)
        
        # Count word frequencies
        word_freq = Counter(all_words)
        
        # Return top N words
        return word_freq.most_common(top_n)
    
    def generate_wordcloud(self, texts, sentiment_filter=None, title="Word Cloud"):
        """Generate a word cloud from texts, optionally filtered by sentiment."""
        all_text = ""
        
        for text in texts:
            if sentiment_filter is not None and self.analyze_sentiment(text)['sentiment'] != sentiment_filter:
                continue
                
            cleaned_text = self.clean_text(text)
            all_text += " " + cleaned_text
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white', 
                             max_words=150,
                             colormap='viridis').generate(all_text)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        
        return plt

class DataProcessor:
    """Class for loading and processing different types of data sources."""
    
    def __init__(self):
        """Initialize the data processor."""
        pass
    
    def load_csv(self, file_path, text_column, date_column=None, additional_columns=None):
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV.")
            
            # Prepare result dataframe
            result_df = pd.DataFrame()
            result_df['text'] = df[text_column]
            
            # Add date column if provided
            if date_column and date_column in df.columns:
                result_df['date'] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Add any additional columns if needed
            if additional_columns:
                for col in additional_columns:
                    if col in df.columns:
                        result_df[col] = df[col]
            
            return result_df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None

    def load_text_file(self, file_path, delimiter='\n'):
        """Load data from a plain text file with one text entry per line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                texts = file.read().split(delimiter)
            
            return pd.DataFrame({'text': texts})
            
        except Exception as e:
            print(f"Error loading text file: {e}")
            return None

class SentimentVisualizer:
    """Class for visualizing sentiment analysis results."""
    
    def __init__(self):
        """Initialize the visualizer."""
        # Set default style
        sns.set(style="whitegrid")
    
    def plot_sentiment_distribution(self, results, title="Sentiment Distribution"):
        """Plot overall sentiment distribution."""
        sentiments = [result['sentiment'] for result in results]
        sentiment_counts = Counter(sentiments)
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Sentiment': list(sentiment_counts.keys()),
            'Count': list(sentiment_counts.values())
        })
        
        # Sort by sentiment category
        sentiment_order = ['positive', 'neutral', 'negative']
        df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=sentiment_order, ordered=True)
        df = df.sort_values('Sentiment')
        
        # Define colors
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        
        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Sentiment', y='Count', data=df, palette=[colors[s] for s in df['Sentiment']])
        
        # Add count labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{int(p.get_height())}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom')
        
        plt.title(title, fontsize=15)
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Sentiment', fontsize=12)
        
        return plt
    
    def plot_sentiment_scores(self, results, title="Sentiment Scores Distribution"):
        """Plot distribution of positive, negative, and neutral scores."""
        # Extract scores
        positive_scores = [result['positive'] for result in results]
        neutral_scores = [result['neutral'] for result in results]
        negative_scores = [result['negative'] for result in results]
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Positive': positive_scores,
            'Neutral': neutral_scores,
            'Negative': negative_scores
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Create box plots
        ax = sns.boxplot(data=df, palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
        
        plt.title(title, fontsize=15)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Sentiment Category', fontsize=12)
        
        return plt
    
    def plot_sentiment_over_time(self, df, title="Sentiment Trends Over Time"):
        """Plot sentiment trends over time if date column is available."""
        if 'date' not in df.columns or 'compound' not in df.columns:
            print("Cannot plot time trends: missing date or compound columns")
            return None
        
        # Ensure date is datetime type
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and calculate average compound score
        daily_sentiment = df.groupby(df['date'].dt.date)['compound'].mean().reset_index()
        
        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(daily_sentiment['date'], daily_sentiment['compound'], marker='o', linestyle='-', color='blue')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add color regions for positive and negative sentiment
        plt.fill_between(daily_sentiment['date'], daily_sentiment['compound'], 0, 
                        where=(daily_sentiment['compound'] >= 0), 
                        color='green', alpha=0.3, label='Positive')
        plt.fill_between(daily_sentiment['date'], daily_sentiment['compound'], 0, 
                        where=(daily_sentiment['compound'] < 0), 
                        color='red', alpha=0.3, label='Negative')
        
        plt.title(title, fontsize=15)
        plt.ylabel('Average Compound Sentiment Score', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt

class InsightGenerator:
    """Class for generating insights from sentiment analysis results."""
    
    def __init__(self):
        """Initialize the insight generator."""
        pass
    
    def generate_summary(self, results, texts):
        """Generate a summary of sentiment analysis results."""
        # Count sentiments
        sentiments = [result['sentiment'] for result in results]
        sentiment_counts = Counter(sentiments)
        total = len(results)
        
        # Calculate percentages
        percentages = {k: round(v/total*100, 1) for k, v in sentiment_counts.items()}
        
        # Calculate average compound score
        avg_compound = round(sum(result['compound'] for result in results) / total, 3)
        
        # Generate summary text
        summary = {
            'total_texts': total,
            'sentiment_distribution': dict(sentiment_counts),
            'sentiment_percentages': percentages,
            'average_compound_score': avg_compound,
        }
        
        # Generate text insights
        insights = []
        
        if percentages.get('positive', 0) > percentages.get('negative', 0):
            insights.append(f"Overall sentiment is positive ({percentages.get('positive', 0)}% positive vs {percentages.get('negative', 0)}% negative).")
        elif percentages.get('negative', 0) > percentages.get('positive', 0):
            insights.append(f"Overall sentiment is negative ({percentages.get('negative', 0)}% negative vs {percentages.get('positive', 0)}% positive).")
        else:
            insights.append(f"Sentiment is balanced ({percentages.get('positive', 0)}% positive vs {percentages.get('negative', 0)}% negative).")
        
        if avg_compound > 0.25:
            insights.append("The average sentiment is strongly positive.")
        elif avg_compound < -0.25:
            insights.append("The average sentiment is strongly negative.")
        elif -0.05 <= avg_compound <= 0.05:
            insights.append("The average sentiment is neutral.")
        
        summary['text_insights'] = insights
        
        return summary
    
    def identify_extreme_sentiments(self, results, texts, n=5):
        """Identify texts with the most extreme positive and negative sentiments."""
        # Create a list of (compound_score, text) tuples
        scored_texts = [(results[i]['compound'], texts[i]) for i in range(len(texts))]
        
        # Sort by compound score
        scored_texts.sort(key=lambda x: x[0])
        
        # Get top N negative and positive texts
        most_negative = scored_texts[:n]
        most_positive = scored_texts[-n:][::-1]  # Reverse to get highest first
        
        return {
            'most_positive': most_positive,
            'most_negative': most_negative
        }

def main():
    """Main function to run the sentiment analysis."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Tool')
    parser.add_argument('--input', type=str, required=True, help='Input file path (CSV or TXT)')
    parser.add_argument('--type', type=str, default='csv', choices=['csv', 'txt'], help='Input file type')
    parser.add_argument('--text-col', type=str, default='text', help='Column name containing text (for CSV)')
    parser.add_argument('--date-col', type=str, default=None, help='Column name containing dates (for CSV)')
    parser.add_argument('--output', type=str, default='sentiment_results.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize classes
    analyzer = SentimentAnalyzer()
    processor = DataProcessor()
    visualizer = SentimentVisualizer()
    insight_gen = InsightGenerator()
    
    # Load data
    if args.type == 'csv':
        df = processor.load_csv(args.input, args.text_col, args.date_col)
    else:
        df = processor.load_text_file(args.input)
    
    if df is None or df.empty:
        print("Error: No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(df)} texts for analysis.")
    
    # Run sentiment analysis
    print("Analyzing sentiment...")
    results = analyzer.analyze_batch(df['text'].tolist())
    
    # Add results to dataframe
    df['compound'] = [result['compound'] for result in results]
    df['positive'] = [result['positive'] for result in results]
    df['neutral'] = [result['neutral'] for result in results]
    df['negative'] = [result['negative'] for result in results]
    df['sentiment'] = [result['sentiment'] for result in results]
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Generate insights
    summary = insight_gen.generate_summary(results, df['text'].tolist())
    extreme = insight_gen.identify_extreme_sentiments(results, df['text'].tolist())
    
    print("\n=== SENTIMENT ANALYSIS SUMMARY ===")
    print(f"Total texts analyzed: {summary['total_texts']}")
    print(f"Sentiment distribution: {summary['sentiment_percentages']}%")
    print(f"Average compound score: {summary['average_compound_score']}")
    
    print("\n=== INSIGHTS ===")
    for insight in summary['text_insights']:
        print(f"- {insight}")
    
    print("\n=== MOST POSITIVE TEXTS ===")
    for score, text in extreme['most_positive']:
        print(f"[{score:.2f}] {text[:100]}...")
    
    print("\n=== MOST NEGATIVE TEXTS ===")
    for score, text in extreme['most_negative']:
        print(f"[{score:.2f}] {text[:100]}...")
    
    print("\n=== EXTRACTING KEYWORDS ===")
    pos_keywords = analyzer.extract_keywords(df['text'].tolist(), 'positive', 10)
    neg_keywords = analyzer.extract_keywords(df['text'].tolist(), 'negative', 10)
    
    print("Top positive keywords:")
    for word, count in pos_keywords:
        print(f"- {word}: {count}")
    
    print("\nTop negative keywords:")
    for word, count in neg_keywords:
        print(f"- {word}: {count}")
    
    # Create visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Sentiment distribution
    plt_dist = visualizer.plot_sentiment_distribution(results, "Sentiment Distribution")
    plt_dist.savefig('sentiment_distribution.png')
    
    # Sentiment scores
    plt_scores = visualizer.plot_sentiment_scores(results)
    plt_scores.savefig('sentiment_scores.png')
    
    # Word clouds
    plt_wc_pos = analyzer.generate_wordcloud(df['text'].tolist(), 'positive', "Positive Sentiment Words")
    plt_wc_pos.savefig('positive_wordcloud.png')
    
    plt_wc_neg = analyzer.generate_wordcloud(df['text'].tolist(), 'negative', "Negative Sentiment Words")
    plt_wc_neg.savefig('negative_wordcloud.png')
    
    # Time trends (if date column exists)
    if 'date' in df.columns:
        plt_time = visualizer.plot_sentiment_over_time(df)
        if plt_time:
            plt_time.savefig('sentiment_trends.png')
    
    print("Visualizations saved as PNG files.")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()