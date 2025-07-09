# Claude 3 Sonnet Powered Exploratory Data Analysis
# Jupyter Notebook for Intelligent EDA with AI Insights

# Cell 1: Installation and Setup
"""
First, install required packages:
!pip install boto3 pandas numpy matplotlib seaborn plotly jupyter-widgets ipywidgets
!pip install textblob wordcloud scikit-learn langchain-aws
"""

# Cell 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# AWS and AI libraries
import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# System libraries
import os
import json
from datetime import datetime
import base64
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# Cell 3: AWS Bedrock Claude Setup
class ClaudeEDAAnalyzer:
    """
    Advanced EDA analyzer powered by Claude 3 Sonnet via AWS Bedrock
    Combines traditional statistical analysis with AI-powered insights
    """
    
    def __init__(self, region_name="us-east-1", aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None):
        """
        Initialize AWS Bedrock Claude client
        
        Args:
            region_name (str): AWS region name
            aws_access_key_id (str): AWS access key ID (optional if using IAM roles/env vars)
            aws_secret_access_key (str): AWS secret access key (optional if using IAM roles/env vars)
            aws_session_token (str): AWS session token (optional, for temporary credentials)
        """
        
        try:
            # Initialize memory saver
            from langchain_core.runnables.utils import Input, Output
            from langchain_core.runnables import RunnableConfig
            
            class MemorySaver:
                def __init__(self):
                    self.memory = {}
                
                def get(self, config):
                    return self.memory.get(config.get("configurable", {}).get("thread_id", "default"), {})
                
                def put(self, config, checkpoint):
                    thread_id = config.get("configurable", {}).get("thread_id", "default")
                    self.memory[thread_id] = checkpoint
            
            memory = MemorySaver()
            
            # Set up AWS credentials if provided
            session_kwargs = {}
            if aws_access_key_id:
                session_kwargs['aws_access_key_id'] = aws_access_key_id
            if aws_secret_access_key:
                session_kwargs['aws_secret_access_key'] = aws_secret_access_key
            if aws_session_token:
                session_kwargs['aws_session_token'] = aws_session_token
            
            # Create boto3 session
            if session_kwargs:
                session = boto3.Session(region_name=region_name, **session_kwargs)
            else:
                session = boto3.Session(region_name=region_name)
            
            # Initialize Claude client via Bedrock
            self.llm = ChatBedrockConverse(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                provider="anthropic",
            )
            
            print("✅ Claude via AWS Bedrock initialized successfully!")
            print(f"📍 Region: {region_name}")
            print(f"🤖 Model: anthropic.claude-3-sonnet-20240229-v1:0")
            self.client_available = True
            
        except Exception as e:
            print(f"❌ Error initializing AWS Bedrock Claude: {str(e)}")
            print("Please ensure you have:")
            print("1. AWS credentials configured (AWS CLI, IAM role, or environment variables)")
            print("2. Bedrock access enabled in your AWS account")
            print("3. Required packages installed: boto3, langchain-aws")
            self.llm = None
            self.client_available = False
            
        self.df = None
        self.analysis_results = {}
        self.ai_insights = {}
    
    def load_data(self, file_path_or_df):
        """Load data from file path or DataFrame"""
        if isinstance(file_path_or_df, str):
            # Load from file
            if file_path_or_df.endswith('.csv'):
                self.df = pd.read_csv(file_path_or_df)
            elif file_path_or_df.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path_or_df)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
        elif isinstance(file_path_or_df, pd.DataFrame):
            # Use existing DataFrame
            self.df = file_path_or_df.copy()
        else:
            raise ValueError("Input must be file path or pandas DataFrame")
            
        print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def get_claude_insights(self, prompt, data_summary=""):
        """
        Get insights from Claude 3 Sonnet via AWS Bedrock
        
        Args:
            prompt (str): The question/prompt for Claude
            data_summary (str): Optional data summary to provide context
        """
        if not self.client_available:
            return "❌ Claude via AWS Bedrock not available. Please configure AWS credentials and Bedrock access."
        
        try:
            full_prompt = f"""
            You are an expert data scientist performing exploratory data analysis. 
            
            Data Context:
            {data_summary}
            
            Question/Task:
            {prompt}
            
            Please provide detailed, actionable insights based on the data. Be specific and practical.
            Focus on patterns, trends, and recommendations that would be valuable for decision-making.
            """
            
            # Create message
            messages = [HumanMessage(content=full_prompt)]
            
            # Get response from Claude
            response = self.llm.invoke(messages)
            
            return response.content
            
        except Exception as e:
            error_msg = f"❌ Error getting Claude insights: {str(e)}"
            print(error_msg)
            return error_msg
    
    def test_connection(self):
        """Test the connection to Claude via AWS Bedrock"""
        if not self.client_available:
            print("❌ Claude client not available")
            return False
        
        try:
            test_response = self.get_claude_insights("Hello, please respond with 'Connection successful' to confirm you're working.")
            print(f"🧪 Test Response: {test_response}")
            return "Connection successful" in test_response or "working" in test_response.lower()
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def generate_data_summary(self):
        """Generate a comprehensive data summary for Claude context"""
        if self.df is None:
            return "No data loaded"
        
        summary = f"""
        Dataset Overview:
        - Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
        - Columns: {', '.join(self.df.columns.tolist())}
        
        Data Types:
        {self.df.dtypes.to_string()}
        
        Missing Values:
        {self.df.isnull().sum().to_string()}
        
        Numerical Columns Summary:
        {self.df.describe().to_string() if len(self.df.select_dtypes(include=[np.number]).columns) > 0 else 'No numerical columns'}
        
        Categorical Columns Sample:
        """
        
        # Add sample values for categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols[:5]:  # Show first 5 categorical columns
            unique_vals = self.df[col].value_counts().head(3)
            summary += f"\n{col}: {unique_vals.to_dict()}"
        
        return summary

# Initialize the analyzer with AWS credentials
# Option 1: Use default AWS credentials (recommended)
analyzer = ClaudeEDAAnalyzer()

# Option 2: Explicitly provide AWS credentials
# analyzer = ClaudeEDAAnalyzer(
#     region_name="us-east-1",
#     aws_access_key_id="YOUR_ACCESS_KEY_ID",
#     aws_secret_access_key="YOUR_SECRET_ACCESS_KEY",
#     aws_session_token="YOUR_SESSION_TOKEN"  # Optional, for temporary credentials
# )

# Test the connection
print("\n🧪 Testing connection to Claude via AWS Bedrock...")
analyzer.test_connection()

# Cell 4: Data Loading and Basic Overview
def load_and_overview(file_path):
    """Load data and get basic overview with Claude insights"""
    
    # Load the data
    df = analyzer.load_data(file_path)
    
    # Display basic information
    print("📊 BASIC DATA OVERVIEW")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display data types
    print(f"\n📋 Column Information:")
    print("-" * 30)
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"{i:2}. {col:<20} | {str(dtype):<10} | Nulls: {null_count} ({null_pct:.1f}%)")
    
    # Show first few rows
    print(f"\n🔍 First 5 rows:")
    display(df.head())
    
    # Get Claude's initial insights
    if analyzer.client:
        print(f"\n🤖 Claude's Initial Data Assessment:")
        print("-" * 40)
        
        data_summary = analyzer.generate_data_summary()
        claude_insights = analyzer.get_claude_insights(
            "What are the key characteristics of this dataset? What potential analysis opportunities do you see? What should I focus on first?",
            data_summary
        )
        
        display(Markdown(claude_insights))
        analyzer.ai_insights['initial_assessment'] = claude_insights
    
    return df

# Example usage:
# df = load_and_overview('your_data.csv')

# Cell 5: Interactive Data Quality Dashboard
def create_data_quality_dashboard(df):
    """Create interactive data quality dashboard"""
    
    # Missing values analysis
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    }).sort_values('Missing_Percentage', ascending=False)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Missing Values by Column', 'Data Type Distribution', 
                       'Missing Values Heatmap', 'Data Completeness'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Missing values bar chart
    fig.add_trace(
        go.Bar(x=missing_data['Column'], y=missing_data['Missing_Percentage'],
               name='Missing %', marker_color='red', opacity=0.7),
        row=1, col=1
    )
    
    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    fig.add_trace(
        go.Pie(labels=dtype_counts.index, values=dtype_counts.values, name="Data Types"),
        row=1, col=2
    )
    
    # Missing values heatmap (sample)
    sample_df = df.head(100)  # Sample for visualization
    missing_matrix = sample_df.isnull().astype(int)
    
    fig.add_trace(
        go.Heatmap(z=missing_matrix.values.T, 
                   y=missing_matrix.columns,
                   colorscale='Reds',
                   showscale=False),
        row=2, col=1
    )
    
    # Completeness score
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    fig.add_trace(
        go.Bar(x=completeness.index, y=completeness.values,
               name='Completeness %', marker_color='green', opacity=0.7),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Data Quality Dashboard")
    fig.show()
    
    # Get Claude's insights on data quality
    if analyzer.client_available:
        print(f"\n🤖 Claude's Data Quality Assessment:")
        print("-" * 40)
        
        quality_prompt = f"""
        Based on this missing values analysis:
        {missing_data.to_string()}
        
        What are the main data quality issues? What recommendations do you have for handling missing values?
        Which columns should I be most concerned about?
        """
        
        quality_insights = analyzer.get_claude_insights(quality_prompt)
        display(Markdown(quality_insights))
        analyzer.ai_insights['data_quality'] = quality_insights
    
    return missing_data

# Cell 6: Smart Categorical Analysis with Claude
def analyze_categorical_with_claude(df, max_categories=10):
    """Analyze categorical variables with Claude's insights"""
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found.")
        return
    
    print(f"📊 CATEGORICAL ANALYSIS ({len(categorical_cols)} columns)")
    print("=" * 50)
    
    # Analyze each categorical column
    for col in categorical_cols:
        print(f"\n🏷️ Column: {col}")
        print("-" * 30)
        
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        print(f"Unique values: {unique_count}")
        
        if unique_count <= max_categories:
            # Show all values for small categories
            print("Value distribution:")
            for value, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {str(value)[:30]:<30}: {count:5} ({pct:5.1f}%)")
            
            # Create visualization
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f'Distribution of {col}')
            fig.update_layout(xaxis_title=col, yaxis_title='Count')
            fig.show()
            
        else:
            # Show top values for large categories
            print(f"Top 10 values (out of {unique_count}):")
            for value, count in value_counts.head(10).items():
                pct = (count / len(df)) * 100
                print(f"  {str(value)[:30]:<30}: {count:5} ({pct:5.1f}%)")
            
            # Create visualization for top values
            top_values = value_counts.head(15)
            fig = px.bar(x=top_values.index, y=top_values.values, 
                        title=f'Top 15 Values in {col}')
            fig.update_layout(xaxis_title=col, yaxis_title='Count')
            fig.show()
        
        # Get Claude's insights for this column
        if analyzer.client_available:
            col_summary = f"""
            Column: {col}
            Unique values: {unique_count}
            Top 5 values: {value_counts.head(5).to_dict()}
            Missing values: {df[col].isnull().sum()}
            """
            
            col_insights = analyzer.get_claude_insights(
                f"What insights can you provide about the '{col}' column? What patterns do you notice? Any recommendations for analysis or data cleaning?",
                col_summary
            )
            
            print(f"\n🤖 Claude's insights on '{col}':")
            display(Markdown(col_insights))
            analyzer.ai_insights[f'categorical_{col}'] = col_insights

# Cell 7: Numerical Analysis with AI Insights
def analyze_numerical_with_claude(df):
    """Analyze numerical variables with Claude's statistical insights"""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found.")
        return
    
    print(f"📈 NUMERICAL ANALYSIS ({len(numerical_cols)} columns)")
    print("=" * 50)
    
    # Statistical summary
    print("\n📊 Statistical Summary:")
    display(df[numerical_cols].describe())
    
    # Create distribution plots
    n_cols = len(numerical_cols)
    fig, axes = plt.subplots((n_cols + 2) // 3, 3, figsize=(15, 5 * ((n_cols + 2) // 3)))
    if n_cols == 1:
        axes = [axes]
    elif n_cols <= 3:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numerical_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_cols == 1:
            ax = axes[0]
        else:
            ax = axes[row, col_idx]
        
        # Create histogram with KDE
        df[col].hist(bins=30, alpha=0.7, ax=ax, edgecolor='black')
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        
        # Add basic statistics as text
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
        ax.legend()
    
    # Remove empty subplots
    for i in range(n_cols, len(axes.flat)):
        fig.delaxes(axes.flat[i])
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        print("\n🔗 Correlation Analysis:")
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.show()
        
        # Get Claude's insights on correlations
        if analyzer.client_available:
            corr_insights = analyzer.get_claude_insights(
                f"Analyze this correlation matrix and identify interesting patterns, strong correlations, and potential relationships: {correlation_matrix.to_string()}",
                f"Numerical columns: {numerical_cols}"
            )
            
            print(f"\n🤖 Claude's Correlation Insights:")
            display(Markdown(corr_insights))
            analyzer.ai_insights['correlation_analysis'] = corr_insights
    
    # Get Claude's overall numerical insights
    if analyzer.client_available:
        num_summary = df[numerical_cols].describe().to_string()
        
        numerical_insights = analyzer.get_claude_insights(
            f"Based on these statistical summaries, what are the key insights about the numerical variables? Any outliers, skewness, or interesting patterns to investigate?",
            f"Statistical summary:\n{num_summary}"
        )
        
        print(f"\n🤖 Claude's Numerical Analysis:")
        display(Markdown(numerical_insights))
        analyzer.ai_insights['numerical_analysis'] = numerical_insights

# Cell 8: Advanced Text Analysis (for comment/text columns)
def analyze_text_with_claude(df, text_column):
    """Advanced text analysis using Claude for insights"""
    
    if text_column not in df.columns:
        print(f"Column '{text_column}' not found in dataset.")
        return
    
    print(f"💬 TEXT ANALYSIS: {text_column}")
    print("=" * 50)
    
    # Basic text statistics
    text_data = df[text_column].dropna().astype(str)
    
    if len(text_data) == 0:
        print("No text data found in this column.")
        return
    
    # Calculate text metrics
    text_lengths = text_data.str.len()
    word_counts = text_data.str.split().str.len()
    
    print(f"Text Statistics:")
    print(f"  Total entries: {len(text_data)}")
    print(f"  Average length: {text_lengths.mean():.1f} characters")
    print(f"  Average words: {word_counts.mean():.1f} words")
    print(f"  Shortest text: {text_lengths.min()} characters")
    print(f"  Longest text: {text_lengths.max()} characters")
    
    # Create text length distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Character length distribution
    axes[0].hist(text_lengths, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Distribution of Text Lengths ({text_column})')
    axes[0].set_xlabel('Characters')
    axes[0].set_ylabel('Frequency')
    
    # Word count distribution
    axes[1].hist(word_counts.dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1].set_title(f'Distribution of Word Counts ({text_column})')
    axes[1].set_xlabel('Words')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Word cloud
    try:
        all_text = ' '.join(text_data.astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {text_column}')
        plt.show()
    except Exception as e:
        print(f"Could not generate word cloud: {e}")
    
    # Sample text analysis with Claude
    if analyzer.client_available and len(text_data) > 0:
        # Get sample texts for Claude analysis
        sample_texts = text_data.sample(min(10, len(text_data))).tolist()
        
        text_analysis_prompt = f"""
        Analyze these sample texts from the '{text_column}' column:
        
        Sample texts:
        {chr(10).join([f"{i+1}. {text[:200]}..." for i, text in enumerate(sample_texts)])}
        
        Text statistics:
        - Total entries: {len(text_data)}
        - Average length: {text_lengths.mean():.1f} characters
        - Average words: {word_counts.mean():.1f} words
        
        Please provide insights on:
        1. Common themes and patterns
        2. Text quality and structure
        3. Potential categories or classifications
        4. Recommendations for further analysis
        """
        
        text_insights = analyzer.get_claude_insights(text_analysis_prompt)
        
        print(f"\n🤖 Claude's Text Analysis Insights:")
        display(Markdown(text_insights))
        analyzer.ai_insights[f'text_analysis_{text_column}'] = text_insights
    
    return text_data

# Cell 9: Interactive Relationship Explorer
def create_relationship_explorer(df):
    """Create interactive widgets to explore relationships between variables"""
    
    # Get column lists
    all_cols = df.columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    def plot_relationship(x_col, y_col, color_col='None', plot_type='scatter'):
        """Plot relationship between two variables"""
        
        if x_col == y_col:
            print("Please select different variables for X and Y axes.")
            return
        
        # Prepare data
        plot_data = df[[x_col, y_col]].dropna()
        
        if color_col != 'None' and color_col in df.columns:
            plot_data[color_col] = df[color_col]
        
        # Create plot based on type
        if plot_type == 'scatter':
            if color_col != 'None':
                fig = px.scatter(plot_data, x=x_col, y=y_col, color=color_col,
                               title=f'{y_col} vs {x_col} (colored by {color_col})')
            else:
                fig = px.scatter(plot_data, x=x_col, y=y_col,
                               title=f'{y_col} vs {x_col}')
                
        elif plot_type == 'box':
            fig = px.box(plot_data, x=x_col, y=y_col,
                        title=f'{y_col} distribution by {x_col}')
                        
        elif plot_type == 'violin':
            fig = px.violin(plot_data, x=x_col, y=y_col,
                           title=f'{y_col} distribution by {x_col}')
        
        fig.show()
        
        # Get Claude's insights on this relationship
        if analyzer.client_available:
            # Calculate basic statistics for the relationship
            if x_col in numerical_cols and y_col in numerical_cols:
                correlation = plot_data[x_col].corr(plot_data[y_col])
                relationship_summary = f"Correlation between {x_col} and {y_col}: {correlation:.3f}"
            else:
                relationship_summary = f"Exploring relationship between {x_col} ({df[x_col].dtype}) and {y_col} ({df[y_col].dtype})"
            
            relationship_prompt = f"""
            I'm exploring the relationship between '{x_col}' and '{y_col}' in my dataset.
            
            Context:
            {relationship_summary}
            
            X variable ({x_col}):
            - Type: {df[x_col].dtype}
            - Sample values: {df[x_col].dropna().head(5).tolist()}
            
            Y variable ({y_col}):
            - Type: {df[y_col].dtype}
            - Sample values: {df[y_col].dropna().head(5).tolist()}
            
            What insights can you provide about this relationship? What should I investigate further?
            """
            
            relationship_insights = analyzer.get_claude_insights(relationship_prompt)
            
            print(f"\n🤖 Claude's Relationship Insights:")
            display(Markdown(relationship_insights))
    
    # Create interactive widgets
    x_widget = widgets.Dropdown(options=all_cols, description='X Variable:')
    y_widget = widgets.Dropdown(options=all_cols, description='Y Variable:')
    color_widget = widgets.Dropdown(options=['None'] + categorical_cols, description='Color by:')
    plot_widget = widgets.Dropdown(
        options=['scatter', 'box', 'violin'], 
        value='scatter',
        description='Plot Type:'
    )
    
    interactive_plot = interactive(plot_relationship, 
                                 x_col=x_widget, 
                                 y_col=y_widget, 
                                 color_col=color_widget,
                                 plot_type=plot_widget)
    
    display(interactive_plot)

# Cell 10: AI-Powered Anomaly Detection
def detect_anomalies_with_claude(df, threshold=3):
    """Detect and analyze anomalies with Claude's insights"""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found for anomaly detection.")
        return
    
    print(f"🔍 ANOMALY DETECTION")
    print("=" * 50)
    
    anomalies_found = {}
    
    for col in numerical_cols:
        # Z-score method for outlier detection
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > threshold]
        
        if len(outliers) > 0:
            anomalies_found[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'values': outliers[col].tolist()
            }
            
            print(f"\n📊 Column: {col}")
            print(f"  Outliers found: {len(outliers)} ({(len(outliers)/len(df)*100):.1f}%)")
            print(f"  Outlier values: {outliers[col].head(10).tolist()}")
            
            # Visualize outliers
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Box plot
            axes[0].boxplot(df[col].dropna())
            axes[0].set_title(f'Box Plot of {col}')
            axes[0].set_ylabel(col)
            
            # Scatter plot with outliers highlighted
            normal_data = df[z_scores <= threshold]
            axes[1].scatter(normal_data.index, normal_data[col], alpha=0.6, label='Normal')
            axes[1].scatter(outliers.index, outliers[col], color='red', alpha=0.8, label='Outliers')
            axes[1].set_title(f'Outliers in {col}')
            axes[1].set_xlabel('Index')
            axes[1].set_ylabel(col)
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()
    
    # Get Claude's insights on anomalies
    if analyzer.client_available and anomalies_found:
        anomaly_summary = ""
        for col, info in anomalies_found.items():
            anomaly_summary += f"\n{col}: {info['count']} outliers ({info['percentage']:.1f}%)"
        
        anomaly_insights = analyzer.get_claude_insights(
            f"I found these outliers in my dataset: {anomaly_summary}\n\nWhat could be causing these anomalies? Should I remove them or investigate further? What's the best approach for handling these outliers?",
            f"Dataset shape: {df.shape}\nNumerical columns analyzed: {numerical_cols}"
        )
        
        print(f"\n🤖 Claude's Anomaly Analysis:")
        display(Markdown(anomaly_insights))
        analyzer.ai_insights['anomaly_detection'] = anomaly_insights
    
    return anomalies_found

# Cell 11: Generate Comprehensive AI Report
def generate_comprehensive_report():
    """Generate a comprehensive EDA report with all Claude insights"""
    
    if not analyzer.ai_insights:
        print("No AI insights available. Please run the analysis functions first.")
        return
    
    print("📋 GENERATING COMPREHENSIVE AI-POWERED EDA REPORT")
    print("=" * 60)
    
    # Create HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Powered EDA Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; }}
            h2 {{ color: #A23B72; margin-top: 30px; }}
            .insight-box {{ 
                background-color: #f8f9fa; 
                border-left: 4px solid #2E86AB; 
                padding: 15px; 
                margin: 20px 0; 
            }}
            .timestamp {{ color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>🤖 AI-Powered Exploratory Data Analysis Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Dataset Shape:</strong> {analyzer.df.shape if analyzer.df is not None else 'N/A'}</p>
    """
    
    # Add each AI insight section
    section_titles = {
        'initial_assessment': '🔍 Initial Data Assessment',
        'data_quality': '🔧 Data Quality Analysis',
        'numerical_analysis': '📈 Numerical Variables Analysis',
        'correlation_analysis': '🔗 Correlation Insights',
        'anomaly_detection': '🚨 Anomaly Detection Results'
    }
    
    for key, insight in analyzer.ai_insights.items():
        if key.startswith('categorical_'):
            col_name = key.replace('categorical_', '')
            title = f"🏷️ Categorical Analysis: {col_name}"
        elif key.startswith('text_analysis_'):
            col_name = key.replace('text_analysis_', '')
            title = f"💬 Text Analysis: {col_name}"
        else:
            title = section_titles.get(key, f"📊 {key.replace('_', ' ').title()}")
        
        html_report += f"""
        <h2>{title}</h2>
        <div class="insight-box">
            {insight.replace(chr(10), '<br>')}
        </div>
        """
    
    html_report += """
        <h2>📝 Summary and Recommendations</h2>
        <div class="insight-box">
            <p>This report was generated using Claude 3 Sonnet AI to provide intelligent insights 
            throughout the exploratory data analysis process. Each section contains AI-powered 
            observations and recommendations based on statistical analysis and pattern recognition.</p>
            
            <p><strong>Next Steps:</strong></p>
            <ul>
                <li>Review flagged data quality issues and implement cleaning strategies</li>
                <li>Investigate interesting correlations and relationships identified</li>
                <li>Address anomalies based on AI recommendations</li>
                <li>Consider advanced modeling based on discovered patterns</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ai_eda_report_{timestamp}.html"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"✅ Comprehensive report saved as: {report_filename}")
    
    # Display summary in notebook
    display(HTML(f"""
    <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2E86AB;">
        <h3>🎯 Key AI Insights Summary</h3>
        <p><strong>Total Insights Generated:</strong> {len(analyzer.ai_insights)}</p>
        <p><strong>Report File:</strong> {report_filename}</p>
        <p><strong>Analysis Completed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """))
    
    return report_filename

# Cell 12: Smart Question Generator
def generate_smart_questions():
    """Generate intelligent follow-up questions based on the dataset"""
    
    if not analyzer.client_available or analyzer.df is None:
        print("❌ Claude client or dataset not available.")
        return
    
    print("❓ GENERATING SMART FOLLOW-UP QUESTIONS")
    print("=" * 50)
    
    # Generate data context for questions
    data_context = f"""
    Dataset Overview:
    - Shape: {analyzer.df.shape}
    - Columns: {', '.join(analyzer.df.columns[:10])}{'...' if len(analyzer.df.columns) > 10 else ''}
    - Data types: {analyzer.df.dtypes.value_counts().to_dict()}
    - Missing values: {analyzer.df.isnull().sum().sum()} total
    
    Sample data preview:
    {analyzer.df.head(3).to_string()}
    """
    
    question_prompt = f"""
    Based on this dataset structure and content, generate 10 intelligent, specific questions 
    that would help uncover deeper insights. Focus on:
    1. Business questions that could be answered with this data
    2. Data quality investigations needed
    3. Relationship explorations between variables
    4. Predictive modeling opportunities
    5. Segmentation and clustering possibilities
    
    Make the questions specific to the actual columns and data types present.
    
    Data context:
    {data_context}
    """
    
    smart_questions = analyzer.get_claude_insights(question_prompt)
    
    print("🤖 Claude's Suggested Analysis Questions:")
    display(Markdown(smart_questions))
    
    # Create interactive question explorer
    def explore_question(question_text):
        """Explore a specific question with Claude"""
        if not question_text.strip():
            return
            
        exploration_prompt = f"""
        Help me explore this question about my dataset: "{question_text}"
        
        Dataset context:
        {data_context}
        
        Please provide:
        1. Specific analysis steps to answer this question
        2. Which columns/variables to focus on
        3. What visualizations would be helpful
        4. Any statistical tests or methods to use
        5. Potential challenges or limitations
        """
        
        exploration_result = analyzer.get_claude_insights(exploration_prompt)
        
        print(f"\n🔍 Analysis Plan for: '{question_text}'")
        print("-" * 60)
        display(Markdown(exploration_result))
    
    # Create text widget for question exploration
    question_input = widgets.Textarea(
        placeholder="Enter a question about your data...",
        description="Question:",
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    explore_button = widgets.Button(description="Explore Question", button_style='primary')
    
    def on_explore_click(b):
        explore_question(question_input.value)
    
    explore_button.on_click(on_explore_click)
    
    print(f"\n💡 Question Explorer:")
    display(widgets.VBox([question_input, explore_button]))
    
    return smart_questions

# Cell 13: Advanced Visualization Generator
def create_advanced_visualizations():
    """Create advanced visualizations with Claude's recommendations"""
    
    if analyzer.df is None:
        print("❌ No dataset loaded.")
        return
    
    print("📊 ADVANCED VISUALIZATION GENERATOR")
    print("=" * 50)
    
    # Get visualization recommendations from Claude
    if analyzer.client_available:
        viz_context = f"""
        Dataset info:
        - Shape: {analyzer.df.shape}
        - Columns: {list(analyzer.df.columns)}
        - Data types: {analyzer.df.dtypes.to_dict()}
        - Numerical columns: {analyzer.df.select_dtypes(include=[np.number]).columns.tolist()}
        - Categorical columns: {analyzer.df.select_dtypes(include=['object']).columns.tolist()}
        """
        
        viz_recommendations = analyzer.get_claude_insights(
            "Recommend 5-7 specific, insightful visualizations for this dataset. For each recommendation, specify the chart type, variables to use, and what insights it would reveal.",
            viz_context
        )
        
        print("🤖 Claude's Visualization Recommendations:")
        display(Markdown(viz_recommendations))
    
    # Interactive dashboard creator
    def create_custom_dashboard():
        """Create a custom dashboard based on user selections"""
        
        numerical_cols = analyzer.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = analyzer.df.select_dtypes(include=['object']).columns.tolist()
        all_cols = analyzer.df.columns.tolist()
        
        # Create dashboard with multiple plots
        if len(numerical_cols) >= 2 and len(categorical_cols) >= 1:
            
            # 1. Correlation heatmap
            if len(numerical_cols) > 1:
                fig_corr = px.imshow(
                    analyzer.df[numerical_cols].corr(),
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                fig_corr.show()
            
            # 2. Distribution comparison
            if len(categorical_cols) > 0 and len(numerical_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numerical_cols[0]
                
                fig_dist = px.box(
                    analyzer.df, 
                    x=cat_col, 
                    y=num_col,
                    title=f"Distribution of {num_col} by {cat_col}"
                )
                fig_dist.show()
            
            # 3. Multi-dimensional scatter plot
            if len(numerical_cols) >= 2:
                scatter_data = analyzer.df.dropna(subset=numerical_cols[:3])
                
                if len(categorical_cols) > 0:
                    color_col = categorical_cols[0]
                    fig_scatter = px.scatter(
                        scatter_data,
                        x=numerical_cols[0],
                        y=numerical_cols[1],
                        size=numerical_cols[2] if len(numerical_cols) >= 3 else None,
                        color=color_col,
                        title=f"Multi-dimensional Analysis"
                    )
                else:
                    fig_scatter = px.scatter(
                        scatter_data,
                        x=numerical_cols[0],
                        y=numerical_cols[1],
                        size=numerical_cols[2] if len(numerical_cols) >= 3 else None,
                        title=f"Relationship between {numerical_cols[0]} and {numerical_cols[1]}"
                    )
                fig_scatter.show()
    
    create_custom_dashboard()

# Cell 14: Model Recommendation Engine
def recommend_models_with_claude():
    """Get machine learning model recommendations from Claude"""
    
    if not analyzer.client_available or analyzer.df is None:
        print("❌ Claude client or dataset not available.")
        return
    
    print("🤖 AI MODEL RECOMMENDATION ENGINE")
    print("=" * 50)
    
    # Analyze dataset characteristics for model recommendations
    numerical_cols = analyzer.df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = analyzer.df.select_dtypes(include=['object']).columns.tolist()
    
    dataset_profile = f"""
    Dataset Characteristics:
    - Total rows: {len(analyzer.df)}
    - Total columns: {len(analyzer.df.columns)}
    - Numerical columns: {len(numerical_cols)} - {numerical_cols}
    - Categorical columns: {len(categorical_cols)} - {categorical_cols}
    - Missing values: {analyzer.df.isnull().sum().sum()} total
    - Data types: {analyzer.df.dtypes.value_counts().to_dict()}
    
    Sample data:
    {analyzer.df.head(3).to_string()}
    
    Statistical summary of numerical columns:
    {analyzer.df[numerical_cols].describe().to_string() if numerical_cols else 'No numerical columns'}
    """
    
    model_recommendation_prompt = f"""
    Based on this dataset profile, please recommend appropriate machine learning approaches:
    
    {dataset_profile}
    
    Please provide:
    1. Specific model types that would work well (regression, classification, clustering, etc.)
    2. Potential target variables for supervised learning
    3. Feature engineering suggestions
    4. Data preprocessing steps needed
    5. Evaluation metrics to use
    6. Potential challenges and how to address them
    7. Whether unsupervised learning approaches might be valuable
    
    Be specific about which algorithms to try and why they're suitable for this data.
    """
    
    model_recommendations = analyzer.get_claude_insights(model_recommendation_prompt)
    
    print("🎯 AI Model Recommendations:")
    display(Markdown(model_recommendations))
    
    # Store recommendations
    analyzer.ai_insights['model_recommendations'] = model_recommendations
    
    return model_recommendations

# Cell 15: Complete EDA Pipeline
def run_complete_eda_pipeline(file_path):
    """
    Run the complete EDA pipeline with Claude 3 Sonnet insights
    
    Args:
        file_path (str): Path to your CSV or Excel file
    """
    
    print("🚀 STARTING COMPLETE AI-POWERED EDA PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Load and overview
        print("\n📂 Step 1: Loading and Basic Overview")
        df = load_and_overview(file_path)
        
        # Step 2: Data quality assessment
        print("\n🔍 Step 2: Data Quality Assessment")
        missing_data = create_data_quality_dashboard(df)
        
        # Step 3: Categorical analysis
        print("\n🏷️ Step 3: Categorical Analysis")
        analyze_categorical_with_claude(df)
        
        # Step 4: Numerical analysis
        print("\n📈 Step 4: Numerical Analysis")
        analyze_numerical_with_claude(df)
        
        # Step 5: Text analysis (if applicable)
        text_columns = [col for col in df.columns if 'comment' in col.lower() or 'text' in col.lower() or 'description' in col.lower()]
        if text_columns:
            print(f"\n💬 Step 5: Text Analysis")
            for text_col in text_columns[:2]:  # Analyze first 2 text columns
                analyze_text_with_claude(df, text_col)
        
        # Step 6: Anomaly detection
        print("\n🔍 Step 6: Anomaly Detection")
        anomalies = detect_anomalies_with_claude(df)
        
        # Step 7: Advanced visualizations
        print("\n📊 Step 7: Advanced Visualizations")
        create_advanced_visualizations()
        
        # Step 8: Model recommendations
        print("\n🤖 Step 8: Model Recommendations")
        recommend_models_with_claude()
        
        # Step 9: Generate smart questions
        print("\n❓ Step 9: Smart Question Generation")
        generate_smart_questions()
        
        # Step 10: Generate comprehensive report
        print("\n📋 Step 10: Generating Comprehensive Report")
        report_file = generate_comprehensive_report()
        
        # Final summary
        print(f"\n✅ COMPLETE EDA PIPELINE FINISHED!")
        print("=" * 60)
        print(f"📊 Dataset analyzed: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"🤖 AI insights generated: {len(analyzer.ai_insights)}")
        print(f"📄 Report saved: {report_file}")
        print(f"⏰ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create interactive relationship explorer
        print(f"\n🔗 Interactive Relationship Explorer:")
        create_relationship_explorer(df)
        
        return df, analyzer.ai_insights
        
    except Exception as e:
        print(f"❌ Error in EDA pipeline: {str(e)}")
        return None, {}

# Cell 16: Quick Start Guide and Examples
print("""
🎯 CLAUDE 3 SONNET VIA AWS BEDROCK - QUICK START GUIDE
================================================================

1. SETUP AWS CREDENTIALS:
   Option A: AWS CLI
   aws configure
   
   Option B: Environment variables
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   export AWS_DEFAULT_REGION="us-east-1"
   
   Option C: IAM Role (recommended for EC2/Lambda)
   
   Option D: Pass directly in code
   analyzer = ClaudeEDAAnalyzer(
       aws_access_key_id="your_key",
       aws_secret_access_key="your_secret",
       region_name="us-east-1"
   )

2. ENSURE BEDROCK ACCESS:
   - Enable Amazon Bedrock in your AWS account
   - Request access to Claude 3 Sonnet model
   - Ensure proper IAM permissions for Bedrock

3. RUN COMPLETE ANALYSIS:
   df, insights = run_complete_eda_pipeline('your_data.csv')

4. RUN INDIVIDUAL ANALYSES:
   # Basic overview
   df = load_and_overview('your_data.csv')
   
   # Data quality
   create_data_quality_dashboard(df)
   
   # Categorical analysis
   analyze_categorical_with_claude(df)
   
   # Numerical analysis
   analyze_numerical_with_claude(df)
   
   # Text analysis
   analyze_text_with_claude(df, 'comment_column')
   
   # Generate questions
   generate_smart_questions()

5. INTERACTIVE FEATURES:
   # Relationship explorer
   create_relationship_explorer(df)
   
   # Custom visualizations
   create_advanced_visualizations()

6. GET CUSTOM INSIGHTS:
   # Ask Claude specific questions
   custom_insight = analyzer.get_claude_insights(
       "What patterns do you see in this data?",
       analyzer.generate_data_summary()
   )

================================================================
🚀 READY TO START! Replace 'your_data.csv' with your file path
================================================================

⚠️  IMPORTANT NOTES:
- Ensure you have AWS Bedrock access enabled
- Claude 3 Sonnet must be available in your region
- Check AWS costs for Bedrock usage
- Test connection before running full analysis
""")

# Example usage cell - uncomment and modify as needed:
"""
# EXAMPLE USAGE:

# 1. Initialize with AWS credentials (automatic detection)
analyzer = ClaudeEDAAnalyzer()

# Or with explicit credentials:
# analyzer = ClaudeEDAAnalyzer(
#     region_name="us-east-1",
#     aws_access_key_id="YOUR_ACCESS_KEY_ID",
#     aws_secret_access_key="YOUR_SECRET_ACCESS_KEY"
# )

# 2. Test connection
analyzer.test_connection()

# 3. Run complete analysis
df, insights = run_complete_eda_pipeline('your_data.csv')

# 4. Or run step by step:
# df = load_and_overview('your_data.csv')
# create_data_quality_dashboard(df)
# analyze_categorical_with_claude(df)
# analyze_numerical_with_claude(df)

# 5. Generate custom insights
# custom_question = "What are the most important variables for predicting customer churn?"
# insight = analyzer.get_claude_insights(custom_question, analyzer.generate_data_summary())
# display(Markdown(insight))
""" your file path
================================================================
""")

# Example usage cell - uncomment and modify as needed:
"""
# EXAMPLE USAGE:

# 1. Initialize with your API key
analyzer = ClaudeEDAAnalyzer()  # Uses environment variable
# or
# analyzer = ClaudeEDAAnalyzer(api_key="your_key_here")

# 2. Run complete analysis
df, insights = run_complete_eda_pipeline('your_data.csv')

# 3. Or run step by step:
# df = load_and_overview('your_data.csv')
# create_data_quality_dashboard(df)
# analyze_categorical_with_claude(df)
# analyze_numerical_with_claude(df)

# 4. Generate custom insights
# custom_question = "What are the most important variables for predicting customer churn?"
# insight = analyzer.get_claude_insights(custom_question, analyzer.generate_data_summary())
# display(Markdown(insight))
"""
