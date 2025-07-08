def site_facility_analysis(self):
        """
        Analyze site and facility patterns
        """
        self.log_text("\n" + "="*60)
        self.log_text("🏢 SITE & FACILITY ANALYSIS")
        self.log_text("="*60)
        
        if 'SITE_NAME' in self.df.columns:
            site_counts = self.df['SITE_NAME'].value_counts()
            self.log_text(f"Total unique sites: {len(site_counts)}")
            self.log_text("\nTop Sites by Record Count:")
            for i, (site, count) in enumerate(site_counts.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                self.log_text(f"{i:2}. {site:<20} | {count:4} records ({pct:5.1f}%)")
        
        if 'FACILITY_AREA' in self.df.columns:
            facility_counts = self.df['FACILITY_AREA'].value_counts()
            self.log_text(f"\nTotal unique facility areas: {len(facility_counts)}")
            self.log_text("\nTop Facility Areas:")
            for i, (facility, count) in enumerate(facility_counts.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                self.log_text(f"{i:2}. {facility:<25} | {count:4} records ({pct:5.1f}%)")
        
        # Site-Facility cross-analysis
        if 'SITE_NAME' in self.df.columns and 'FACILITY_AREA' in self.df.columns:
            self.log_text(f"\nSite-Facility Combinations:")
            site_facility = self.df.groupby(['SITE_NAME', 'FACILITY_AREA']).size().sort_valuesimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
from datetime import datetime
import io
import sys
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CSVExploratoryAnalysis:
    """
    Comprehensive EDA class for risk assessment/compliance data
    Saves all outputs to local 'analysis' folder
    """
    
    def __init__(self, file_path):
        """
        Initialize with CSV file path and create analysis folder
        """
        self.file_path = file_path
        self.df = None
        self.analysis_folder = "analysis"
        self.text_output = []
        self.chart_counter = 1
        
        # Create analysis folder
        self.create_analysis_folder()
        
        # Initialize text output file
        self.text_file_path = os.path.join(self.analysis_folder, "analysis_report.txt")
        
        # Load data
        self.load_data()
    
    def create_analysis_folder(self):
        """
        Create analysis folder if it doesn't exist
        """
        if not os.path.exists(self.analysis_folder):
            os.makedirs(self.analysis_folder)
            print(f"✅ Created '{self.analysis_folder}' folder")
        else:
            print(f"📁 Using existing '{self.analysis_folder}' folder")
    
    def log_text(self, text):
        """
        Log text to both console and text file
        """
        print(text)
        self.text_output.append(text)
    
    def save_text_output(self):
        """
        Save all text output to file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{'='*80}
CSV EXPLORATORY DATA ANALYSIS REPORT
Generated on: {timestamp}
Dataset: {self.file_path}
{'='*80}

"""
        
        with open(self.text_file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write('\n'.join(self.text_output))
        
        print(f"\n💾 Text analysis saved to: {self.text_file_path}")
    
    def save_chart(self, filename_prefix, title=""):
        """
        Save current matplotlib figure to analysis folder
        """
        filename = f"{self.chart_counter:02d}_{filename_prefix}.png"
        filepath = os.path.join(self.analysis_folder, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📊 Chart saved: {filename}")
        self.chart_counter += 1
        
        return filepath
    
    def load_data(self):
        """
        Load and initial data inspection
        """
        try:
            self.df = pd.read_csv(self.file_path)
            self.log_text(f"✅ Data loaded successfully!")
            self.log_text(f"Dataset shape: {self.df.shape}")
        except Exception as e:
            self.log_text(f"❌ Error loading data: {e}")
            return
    
    def basic_info(self):
        """
        Display basic information about the dataset
        """
        self.log_text("\n" + "="*60)
        self.log_text("📊 BASIC DATASET INFORMATION")
        self.log_text("="*60)
        
        self.log_text(f"Shape: {self.df.shape}")
        self.log_text(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        self.log_text("\n📋 Column Information:")
        self.log_text("-" * 40)
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            self.log_text(f"{i:2}. {col:<20} | {str(dtype):<10} | Nulls: {null_count} ({null_pct:.1f}%)")
        
        self.log_text(f"\n📈 Data Types Summary:")
        dtype_summary = self.df.dtypes.value_counts().to_string()
        self.log_text(dtype_summary)
        
        self.log_text(f"\n🔍 First 3 rows:")
        first_rows = self.df.head(3).to_string()
        self.log_text(first_rows)
    
    def data_quality_assessment(self):
        """
        Comprehensive data quality analysis
        """
        self.log_text("\n" + "="*60)
        self.log_text("🔍 DATA QUALITY ASSESSMENT")
        self.log_text("="*60)
        
        # Missing values analysis
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Data_Type': self.df.dtypes
        }).sort_values('Missing_Percentage', ascending=False)
        
        self.log_text("\n📉 Missing Values Summary:")
        self.log_text(missing_data.to_string(index=False))
        
        # Duplicate analysis
        duplicate_count = self.df.duplicated().sum()
        self.log_text(f"\n🔄 Duplicate Rows: {duplicate_count} ({(duplicate_count/len(self.df)*100):.2f}%)")
        
        # Unique values per column
        self.log_text(f"\n🆔 Unique Values per Column:")
        unique_counts = self.df.nunique().sort_values(ascending=False)
        for col, count in unique_counts.items():
            pct = (count / len(self.df)) * 100
            self.log_text(f"{col:<20}: {count:5} unique ({pct:.1f}%)")
        
        # Data quality visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Overview', fontsize=16, fontweight='bold')
        
        # Missing values heatmap
        sns.heatmap(self.df.isnull(), ax=axes[0,0], cbar=True, yticklabels=False, 
                   cmap='viridis', xticklabels=True)
        axes[0,0].set_title('Missing Values Pattern')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Missing values bar plot
        missing_data.set_index('Column')['Missing_Percentage'].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Missing Values by Column (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Unique values distribution
        unique_counts.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Unique Values per Column')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Data completeness
        completeness = (1 - self.df.isnull().sum() / len(self.df)) * 100
        completeness.plot(kind='bar', ax=axes[1,1], color='green', alpha=0.7)
        axes[1,1].set_title('Data Completeness (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylim(0, 100)
        
        plt.tight_layout()
        self.save_chart("data_quality_overview", "Data Quality Analysis")
        plt.show()
    
    def categorical_analysis(self):
        """
        Analyze categorical columns
        """
        self.log_text("\n" + "="*60)
        self.log_text("📊 CATEGORICAL DATA ANALYSIS")
        self.log_text("="*60)
        
        # Identify categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        self.log_text(f"Categorical Columns: {categorical_cols}")
        
        # Analyze each categorical column
        for col in categorical_cols:
            self.log_text(f"\n🏷️ Column: {col}")
            self.log_text("-" * 40)
            
            value_counts = self.df[col].value_counts()
            self.log_text(f"Unique values: {len(value_counts)}")
            self.log_text(f"Most common value: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)")
            
            # Show top 10 values
            self.log_text("\nTop 10 values:")
            top_values = value_counts.head(10)
            for idx, (value, count) in enumerate(top_values.items(), 1):
                pct = (count / len(self.df)) * 100
                self.log_text(f"{idx:2}. {str(value):<30} | {count:4} ({pct:5.1f}%)")
        
        # Visualizations for categorical data
        n_cols = len(categorical_cols)
        if n_cols > 0:
            fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(15, 4 * ((n_cols + 1) // 2)))
            if n_cols == 1:
                axes = [axes]
            elif n_cols <= 2:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(categorical_cols):
                row = i // 2
                col_idx = i % 2
                
                if n_cols == 1:
                    ax = axes[0]
                else:
                    ax = axes[row, col_idx]
                
                # Plot top 15 values
                top_15 = self.df[col].value_counts().head(15)
                top_15.plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.tick_params(axis='x', rotation=45)
                
            # Remove empty subplots
            if n_cols % 2 == 1 and n_cols > 1:
                fig.delaxes(axes[-1, -1])
            
            plt.tight_layout()
            self.save_chart("categorical_distributions", "Categorical Data Distributions")
            plt.show()
    
    def response_analysis(self):
        """
        Specific analysis for RESPONSE column (assuming YES/NO type responses)
        """
        if 'RESPONSE' in self.df.columns:
            self.log_text("\n" + "="*60)
            self.log_text("✅ RESPONSE ANALYSIS")
            self.log_text("="*60)
            
            response_counts = self.df['RESPONSE'].value_counts()
            response_pct = self.df['RESPONSE'].value_counts(normalize=True) * 100
            
            self.log_text("Response Distribution:")
            for response, count in response_counts.items():
                pct = response_pct[response]
                self.log_text(f"{response:<15}: {count:4} ({pct:5.1f}%)")
            
            # Response by other categorical variables
            categorical_cols = ['SITE_NAME', 'FACILITY_AREA', 'AUTHOR_NAME', 'TEMPLATE_NAME']
            available_cols = [col for col in categorical_cols if col in self.df.columns]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Response distribution pie chart
            self.df['RESPONSE'].value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%')
            axes[0].set_title('Overall Response Distribution')
            axes[0].set_ylabel('')
            
            # Response by categorical variables
            for i, col in enumerate(available_cols[:3], 1):
                if i < 4:
                    crosstab = pd.crosstab(self.df[col], self.df['RESPONSE'])
                    crosstab.plot(kind='bar', ax=axes[i], stacked=True)
                    axes[i].set_title(f'Response by {col}')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(title='Response')
            
            plt.tight_layout()
            self.save_chart("response_analysis", "Response Pattern Analysis")
            plt.show()
    
    def author_analysis(self):
        """
        Analyze author patterns and productivity
        """
        if 'AUTHOR_NAME' in self.df.columns:
            self.log_text("\n" + "="*60)
            self.log_text("👤 AUTHOR ANALYSIS")
            self.log_text("="*60)
            
            author_stats = self.df['AUTHOR_NAME'].value_counts()
            self.log_text(f"Total unique authors: {len(author_stats)}")
            self.log_text(f"Most active author: {author_stats.index[0]} ({author_stats.iloc[0]} records)")
            self.log_text(f"Average records per author: {author_stats.mean():.1f}")
            
            self.log_text("\nTop 10 Most Active Authors:")
            for i, (author, count) in enumerate(author_stats.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                self.log_text(f"{i:2}. {author:<25} | {count:4} records ({pct:5.1f}%)")
            
            # Author productivity visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Top 15 authors
            author_stats.head(15).plot(kind='bar', ax=axes[0])
            axes[0].set_title('Top 15 Authors by Record Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Author productivity distribution
            axes[1].hist(author_stats.values, bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_title('Distribution of Records per Author')
            axes[1].set_xlabel('Number of Records')
            axes[1].set_ylabel('Number of Authors')
            
            plt.tight_layout()
            self.save_chart("author_analysis", "Author Productivity Analysis")
            plt.show()
    
    def site_facility_analysis(self):
        """
        Analyze site and facility patterns
        """
        print("\n" + "="*60)
        print("🏢 SITE & FACILITY ANALYSIS")
        print("="*60)
        
        if 'SITE_NAME' in self.df.columns:
            site_counts = self.df['SITE_NAME'].value_counts()
            print(f"Total unique sites: {len(site_counts)}")
            print("\nTop Sites by Record Count:")
            for i, (site, count) in enumerate(site_counts.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"{i:2}. {site:<20} | {count:4} records ({pct:5.1f}%)")
        
        if 'FACILITY_AREA' in self.df.columns:
            facility_counts = self.df['FACILITY_AREA'].value_counts()
            print(f"\nTotal unique facility areas: {len(facility_counts)}")
            print("\nTop Facility Areas:")
            for i, (facility, count) in enumerate(facility_counts.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"{i:2}. {facility:<25} | {count:4} records ({pct:5.1f}%)")
        
        # Site-Facility cross-analysis
        if 'SITE_NAME' in self.df.columns and 'FACILITY_AREA' in self.df.columns:
            self.log_text(f"\nSite-Facility Combinations:")
            site_facility = self.df.groupby(['SITE_NAME', 'FACILITY_AREA']).size().sort_values(ascending=False)
            self.log_text(site_facility.head(10).to_string())
            
            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Site distribution
            if len(site_counts) <= 20:
                site_counts.plot(kind='bar', ax=axes[0,0])
                axes[0,0].tick_params(axis='x', rotation=45)
            else:
                site_counts.head(20).plot(kind='bar', ax=axes[0,0])
                axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].set_title('Records by Site')
            
            # Facility distribution
            if len(facility_counts) <= 20:
                facility_counts.plot(kind='bar', ax=axes[0,1])
                axes[0,1].tick_params(axis='x', rotation=45)
            else:
                facility_counts.head(20).plot(kind='bar', ax=axes[0,1])
                axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].set_title('Records by Facility Area')
            
            # Heatmap of top site-facility combinations
            pivot_data = self.df.pivot_table(index='SITE_NAME', columns='FACILITY_AREA', 
                                           aggfunc='size', fill_value=0)
            # Select top sites and facilities for readability
            top_sites = site_counts.head(10).index
            top_facilities = facility_counts.head(10).index
            
            heatmap_data = pivot_data.loc[top_sites, top_facilities]
            sns.heatmap(heatmap_data, ax=axes[1,0], cmap='YlOrRd', annot=True, fmt='d')
            axes[1,0].set_title('Site-Facility Heatmap (Top 10 each)')
            
            # Distribution of records per site
            axes[1,1].hist(site_counts.values, bins=15, alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Distribution of Records per Site')
            axes[1,1].set_xlabel('Number of Records')
            axes[1,1].set_ylabel('Number of Sites')
            
            plt.tight_layout()
            self.save_chart("site_facility_analysis", "Site and Facility Analysis")
            plt.show()
    
    def template_analysis(self):
        """
        Analyze assessment templates
        """
        if 'TEMPLATE_NAME' in self.df.columns:
            self.log_text("\n" + "="*60)
            self.log_text("📋 TEMPLATE ANALYSIS")
            self.log_text("="*60)
            
            template_counts = self.df['TEMPLATE_NAME'].value_counts()
            self.log_text(f"Total unique templates: {len(template_counts)}")
            
            self.log_text("\nTemplate Usage:")
            for i, (template, count) in enumerate(template_counts.items(), 1):
                pct = (count / len(self.df)) * 100
                self.log_text(f"{i:2}. {str(template)[:50]:<52} | {count:4} ({pct:5.1f}%)")
            
            # Template visualization
            plt.figure(figsize=(12, 8))
            template_counts.plot(kind='barh')
            plt.title('Template Usage Distribution')
            plt.xlabel('Number of Records')
            plt.tight_layout()
            self.save_chart("template_analysis", "Template Usage Analysis")
            plt.show()
    
    def comment_analysis(self):
        """
        Analyze comment patterns and text length
        """
        if 'COMMENT' in self.df.columns:
            self.log_text("\n" + "="*60)
            self.log_text("💬 COMMENT ANALYSIS")
            self.log_text("="*60)
            
            # Basic comment statistics
            comments = self.df['COMMENT'].dropna()
            self.log_text(f"Total comments: {len(comments)}")
            self.log_text(f"Comments with data: {len(comments)} ({len(comments)/len(self.df)*100:.1f}%)")
            
            if len(comments) > 0:
                # Comment length analysis
                comment_lengths = comments.str.len()
                self.log_text(f"\nComment Length Statistics:")
                self.log_text(f"Average length: {comment_lengths.mean():.1f} characters")
                self.log_text(f"Median length: {comment_lengths.median():.1f} characters")
                self.log_text(f"Min length: {comment_lengths.min()}")
                self.log_text(f"Max length: {comment_lengths.max()}")
                
                # Word count analysis
                word_counts = comments.str.split().str.len()
                self.log_text(f"\nWord Count Statistics:")
                self.log_text(f"Average words: {word_counts.mean():.1f}")
                self.log_text(f"Median words: {word_counts.median():.1f}")
                
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Comment length distribution
                axes[0,0].hist(comment_lengths, bins=30, alpha=0.7, edgecolor='black')
                axes[0,0].set_title('Distribution of Comment Lengths (Characters)')
                axes[0,0].set_xlabel('Characters')
                axes[0,0].set_ylabel('Frequency')
                
                # Word count distribution
                axes[0,1].hist(word_counts, bins=20, alpha=0.7, edgecolor='black', color='green')
                axes[0,1].set_title('Distribution of Word Counts')
                axes[0,1].set_xlabel('Words')
                axes[0,1].set_ylabel('Frequency')
                
                # Comment length by response (if available)
                if 'RESPONSE' in self.df.columns:
                    for response in self.df['RESPONSE'].unique():
                        if pd.notna(response):
                            subset_lengths = self.df[self.df['RESPONSE'] == response]['COMMENT'].str.len().dropna()
                            axes[1,0].hist(subset_lengths, alpha=0.6, label=response, bins=20)
                    axes[1,0].set_title('Comment Length by Response Type')
                    axes[1,0].legend()
                    axes[1,0].set_xlabel('Characters')
                    axes[1,0].set_ylabel('Frequency')
                
                # Empty vs filled comments
                comment_status = ['Has Comment', 'No Comment']
                comment_counts = [len(comments), len(self.df) - len(comments)]
                axes[1,1].pie(comment_counts, labels=comment_status, autopct='%1.1f%%')
                axes[1,1].set_title('Comment Availability')
                
                plt.tight_layout()
                self.save_chart("comment_analysis", "Comment Analysis")
                plt.show()
                
                # Show sample long and short comments
                self.log_text(f"\nSample Long Comment ({comment_lengths.max()} chars):")
                long_comment_idx = comment_lengths.idxmax()
                self.log_text(f"'{comments.loc[long_comment_idx][:200]}...'")
                
                self.log_text(f"\nSample Short Comment ({comment_lengths.min()} chars):")
                short_comment_idx = comment_lengths.idxmin()
                self.log_text(f"'{comments.loc[short_comment_idx]}'")
    
    def correlation_analysis(self):
        """
        Analyze correlations between categorical variables
        """
        self.log_text("\n" + "="*60)
        self.log_text("🔗 CORRELATION ANALYSIS")
        self.log_text("="*60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) >= 2:
            self.log_text("Cross-tabulation analysis between key variables:")
            
            # Key relationships to analyze
            relationships = [
                ('SITE_NAME', 'RESPONSE'),
                ('FACILITY_AREA', 'RESPONSE'),
                ('AUTHOR_NAME', 'RESPONSE'),
                ('TEMPLATE_NAME', 'RESPONSE'),
                ('SITE_NAME', 'FACILITY_AREA')
            ]
            
            for col1, col2 in relationships:
                if col1 in self.df.columns and col2 in self.df.columns:
                    self.log_text(f"\n{col1} vs {col2}:")
                    crosstab = pd.crosstab(self.df[col1], self.df[col2])
                    self.log_text(crosstab.head().to_string())
                    
                    # Chi-square test if applicable
                    try:
                        from scipy.stats import chi2_contingency
                        chi2, p_value, dof, expected = chi2_contingency(crosstab)
                        self.log_text(f"Chi-square test p-value: {p_value:.4f}")
                        if p_value < 0.05:
                            self.log_text("Significant association detected!")
                    except:
                        self.log_text("Chi-square test not available")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        self.log_text("\n" + "="*80)
        self.log_text("📊 EXECUTIVE SUMMARY REPORT")
        self.log_text("="*80)
        
        self.log_text(f"Dataset Overview:")
        self.log_text(f"• Total Records: {len(self.df):,}")
        self.log_text(f"• Total Columns: {len(self.df.columns)}")
        self.log_text(f"• Data Quality: {((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100):.1f}% complete")
        
        if 'SITE_NAME' in self.df.columns:
            self.log_text(f"• Unique Sites: {self.df['SITE_NAME'].nunique()}")
        
        if 'FACILITY_AREA' in self.df.columns:
            self.log_text(f"• Unique Facility Areas: {self.df['FACILITY_AREA'].nunique()}")
        
        if 'AUTHOR_NAME' in self.df.columns:
            self.log_text(f"• Unique Authors: {self.df['AUTHOR_NAME'].nunique()}")
        
        if 'TEMPLATE_NAME' in self.df.columns:
            self.log_text(f"• Unique Templates: {self.df['TEMPLATE_NAME'].nunique()}")
        
        if 'RESPONSE' in self.df.columns:
            response_dist = self.df['RESPONSE'].value_counts(normalize=True) * 100
            self.log_text(f"\nResponse Distribution:")
            for response, pct in response_dist.items():
                self.log_text(f"• {response}: {pct:.1f}%")
        
        self.log_text(f"\nData Quality Issues:")
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            self.log_text(f"• Columns with missing data: {', '.join(missing_cols)}")
        else:
            self.log_text("• No missing data detected")
        
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.log_text(f"• Duplicate rows: {duplicates}")
        else:
            self.log_text("• No duplicate rows detected")
    
    def run_full_analysis(self):
        """
        Run the complete EDA pipeline
        """
        if self.df is None:
            self.log_text("❌ No data loaded. Please check file path.")
            return
        
        self.log_text("🚀 Starting Comprehensive EDA Analysis...")
        
        self.basic_info()
        self.data_quality_assessment()
        self.categorical_analysis()
        self.response_analysis()
        self.author_analysis()
        self.site_facility_analysis()
        self.template_analysis()
        self.comment_analysis()
        self.correlation_analysis()
        self.generate_summary_report()
        
        # Save all text output to file
        self.save_text_output()
        
        self.log_text(f"\n✅ Analysis Complete!")
        self.log_text(f"📁 Dataset: {self.file_path}")
        self.log_text(f"📊 Records Analyzed: {len(self.df):,}")
        self.log_text(f"📁 Charts saved to: {self.analysis_folder}/")
        self.log_text(f"📄 Text report saved to: {self.text_file_path}")
        
        # Create summary file with file locations
        self.create_file_summary()

    
    def create_file_summary(self):
        """
        Create a summary file listing all generated files
        """
        summary_path = os.path.join(self.analysis_folder, "file_summary.txt")
        
        # Get list of all files in analysis folder
        analysis_files = os.listdir(self.analysis_folder)
        chart_files = [f for f in analysis_files if f.endswith('.png')]
        
        summary_content = f"""
EDA ANALYSIS FILES SUMMARY
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {self.file_path}

📊 CHARTS GENERATED ({len(chart_files)} files):
{'='*50}
"""
        
        for i, chart_file in enumerate(sorted(chart_files), 1):
            summary_content += f"{i:2}. {chart_file}\n"
        
        summary_content += f"""
📄 TEXT REPORTS:
{'='*50}
1. analysis_report.txt - Complete text analysis
2. file_summary.txt - This file summary

📁 FOLDER STRUCTURE:
{'='*50}
analysis/
├── analysis_report.txt     (Complete text analysis)
├── file_summary.txt        (This summary)
"""
        
        for chart_file in sorted(chart_files):
            summary_content += f"├── {chart_file}\n"
        
        summary_content += f"""
🔧 USAGE:
{'='*50}
- Open analysis_report.txt for complete text analysis
- View PNG files for charts and visualizations
- All files are saved in high resolution (300 DPI)
- Charts are optimized for presentations and reports

📊 ANALYSIS SECTIONS:
{'='*50}
1. Basic Dataset Information
2. Data Quality Assessment
3. Categorical Data Analysis
4. Response Pattern Analysis
5. Author Productivity Analysis
6. Site & Facility Analysis
7. Template Usage Analysis
8. Comment Analysis
9. Correlation Analysis
10. Executive Summary Report
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📋 File summary saved to: {summary_path}")

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual file path
    file_path = "your_data.csv"
    
    # Create analyzer instance
    analyzer = CSVExploratoryAnalysis(file_path)
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    # Or run individual analyses
    # analyzer.basic_info()
    # analyzer.data_quality_assessment()
    # analyzer.response_analysis()
    
    print("\n" + "="*60)
    print("🎯 USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Save this code as 'csv_eda.py'")
    print("2. Install required packages: pip install pandas numpy matplotlib seaborn scipy")
    print("3. Update file_path variable with your CSV file path")
    print("4. Run: python csv_eda.py")
    print("5. For Jupyter: Create analyzer = CSVExploratoryAnalysis('file.csv')")
    print("6. Then run: analyzer.run_full_analysis()")
    print("")
    print("📁 OUTPUT FILES:")
    print("• analysis/analysis_report.txt - Complete text analysis")
    print("• analysis/*.png - All charts and visualizations")
    print("• analysis/file_summary.txt - Summary of all generated files")
    print("")
    print("✨ All outputs are automatically saved to 'analysis' folder!")
