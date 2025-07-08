import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CSVExploratoryAnalysis:
    """
    Comprehensive EDA class for risk assessment/compliance data
    """
    
    def __init__(self, file_path):
        """
        Initialize with CSV file path
        """
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Load and initial data inspection
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    def basic_info(self):
        """
        Display basic information about the dataset
        """
        print("\n" + "="*60)
        print("📊 BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📋 Column Information:")
        print("-" * 40)
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            print(f"{i:2}. {col:<20} | {str(dtype):<10} | Nulls: {null_count} ({null_pct:.1f}%)")
        
        print(f"\n📈 Data Types Summary:")
        print(self.df.dtypes.value_counts())
        
        print(f"\n🔍 First 3 rows:")
        print(self.df.head(3).to_string())
    
    def data_quality_assessment(self):
        """
        Comprehensive data quality analysis
        """
        print("\n" + "="*60)
        print("🔍 DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Missing values analysis
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Data_Type': self.df.dtypes
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("\n📉 Missing Values Summary:")
        print(missing_data.to_string(index=False))
        
        # Duplicate analysis
        duplicate_count = self.df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {duplicate_count} ({(duplicate_count/len(self.df)*100):.2f}%)")
        
        # Unique values per column
        print(f"\n🆔 Unique Values per Column:")
        unique_counts = self.df.nunique().sort_values(ascending=False)
        for col, count in unique_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"{col:<20}: {count:5} unique ({pct:.1f}%)")
        
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
        plt.show()
    
    def categorical_analysis(self):
        """
        Analyze categorical columns
        """
        print("\n" + "="*60)
        print("📊 CATEGORICAL DATA ANALYSIS")
        print("="*60)
        
        # Identify categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Categorical Columns: {categorical_cols}")
        
        # Analyze each categorical column
        for col in categorical_cols:
            print(f"\n🏷️ Column: {col}")
            print("-" * 40)
            
            value_counts = self.df[col].value_counts()
            print(f"Unique values: {len(value_counts)}")
            print(f"Most common value: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)")
            
            # Show top 10 values
            print("\nTop 10 values:")
            top_values = value_counts.head(10)
            for idx, (value, count) in enumerate(top_values.items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"{idx:2}. {str(value):<30} | {count:4} ({pct:5.1f}%)")
        
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
            plt.show()
    
    def response_analysis(self):
        """
        Specific analysis for RESPONSE column (assuming YES/NO type responses)
        """
        if 'RESPONSE' in self.df.columns:
            print("\n" + "="*60)
            print("✅ RESPONSE ANALYSIS")
            print("="*60)
            
            response_counts = self.df['RESPONSE'].value_counts()
            response_pct = self.df['RESPONSE'].value_counts(normalize=True) * 100
            
            print("Response Distribution:")
            for response, count in response_counts.items():
                pct = response_pct[response]
                print(f"{response:<15}: {count:4} ({pct:5.1f}%)")
            
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
            plt.show()
    
    def author_analysis(self):
        """
        Analyze author patterns and productivity
        """
        if 'AUTHOR_NAME' in self.df.columns:
            print("\n" + "="*60)
            print("👤 AUTHOR ANALYSIS")
            print("="*60)
            
            author_stats = self.df['AUTHOR_NAME'].value_counts()
            print(f"Total unique authors: {len(author_stats)}")
            print(f"Most active author: {author_stats.index[0]} ({author_stats.iloc[0]} records)")
            print(f"Average records per author: {author_stats.mean():.1f}")
            
            print("\nTop 10 Most Active Authors:")
            for i, (author, count) in enumerate(author_stats.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"{i:2}. {author:<25} | {count:4} records ({pct:5.1f}%)")
            
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
            print(f"\nSite-Facility Combinations:")
            site_facility = self.df.groupby(['SITE_NAME', 'FACILITY_AREA']).size().sort_values(ascending=False)
            print(site_facility.head(10).to_string())
            
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
            plt.show()
    
    def template_analysis(self):
        """
        Analyze assessment templates
        """
        if 'TEMPLATE_NAME' in self.df.columns:
            print("\n" + "="*60)
            print("📋 TEMPLATE ANALYSIS")
            print("="*60)
            
            template_counts = self.df['TEMPLATE_NAME'].value_counts()
            print(f"Total unique templates: {len(template_counts)}")
            
            print("\nTemplate Usage:")
            for i, (template, count) in enumerate(template_counts.items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"{i:2}. {str(template)[:50]:<52} | {count:4} ({pct:5.1f}%)")
            
            # Template visualization
            plt.figure(figsize=(12, 8))
            template_counts.plot(kind='barh')
            plt.title('Template Usage Distribution')
            plt.xlabel('Number of Records')
            plt.tight_layout()
            plt.show()
    
    def comment_analysis(self):
        """
        Analyze comment patterns and text length
        """
        if 'COMMENT' in self.df.columns:
            print("\n" + "="*60)
            print("💬 COMMENT ANALYSIS")
            print("="*60)
            
            # Basic comment statistics
            comments = self.df['COMMENT'].dropna()
            print(f"Total comments: {len(comments)}")
            print(f"Comments with data: {len(comments)} ({len(comments)/len(self.df)*100:.1f}%)")
            
            if len(comments) > 0:
                # Comment length analysis
                comment_lengths = comments.str.len()
                print(f"\nComment Length Statistics:")
                print(f"Average length: {comment_lengths.mean():.1f} characters")
                print(f"Median length: {comment_lengths.median():.1f} characters")
                print(f"Min length: {comment_lengths.min()}")
                print(f"Max length: {comment_lengths.max()}")
                
                # Word count analysis
                word_counts = comments.str.split().str.len()
                print(f"\nWord Count Statistics:")
                print(f"Average words: {word_counts.mean():.1f}")
                print(f"Median words: {word_counts.median():.1f}")
                
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
                plt.show()
                
                # Show sample long and short comments
                print(f"\nSample Long Comment ({comment_lengths.max()} chars):")
                long_comment_idx = comment_lengths.idxmax()
                print(f"'{comments.loc[long_comment_idx][:200]}...'")
                
                print(f"\nSample Short Comment ({comment_lengths.min()} chars):")
                short_comment_idx = comment_lengths.idxmin()
                print(f"'{comments.loc[short_comment_idx]}'")
    
    def correlation_analysis(self):
        """
        Analyze correlations between categorical variables
        """
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) >= 2:
            print("Cross-tabulation analysis between key variables:")
            
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
                    print(f"\n{col1} vs {col2}:")
                    crosstab = pd.crosstab(self.df[col1], self.df[col2])
                    print(crosstab.head())
                    
                    # Chi-square test if applicable
                    try:
                        from scipy.stats import chi2_contingency
                        chi2, p_value, dof, expected = chi2_contingency(crosstab)
                        print(f"Chi-square test p-value: {p_value:.4f}")
                        if p_value < 0.05:
                            print("Significant association detected!")
                    except:
                        print("Chi-square test not available")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "="*80)
        print("📊 EXECUTIVE SUMMARY REPORT")
        print("="*80)
        
        print(f"Dataset Overview:")
        print(f"• Total Records: {len(self.df):,}")
        print(f"• Total Columns: {len(self.df.columns)}")
        print(f"• Data Quality: {((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100):.1f}% complete")
        
        if 'SITE_NAME' in self.df.columns:
            print(f"• Unique Sites: {self.df['SITE_NAME'].nunique()}")
        
        if 'FACILITY_AREA' in self.df.columns:
            print(f"• Unique Facility Areas: {self.df['FACILITY_AREA'].nunique()}")
        
        if 'AUTHOR_NAME' in self.df.columns:
            print(f"• Unique Authors: {self.df['AUTHOR_NAME'].nunique()}")
        
        if 'TEMPLATE_NAME' in self.df.columns:
            print(f"• Unique Templates: {self.df['TEMPLATE_NAME'].nunique()}")
        
        if 'RESPONSE' in self.df.columns:
            response_dist = self.df['RESPONSE'].value_counts(normalize=True) * 100
            print(f"\nResponse Distribution:")
            for response, pct in response_dist.items():
                print(f"• {response}: {pct:.1f}%")
        
        print(f"\nData Quality Issues:")
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            print(f"• Columns with missing data: {', '.join(missing_cols)}")
        else:
            print("• No missing data detected")
        
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"• Duplicate rows: {duplicates}")
        else:
            print("• No duplicate rows detected")
    
    def run_full_analysis(self):
        """
        Run the complete EDA pipeline
        """
        if self.df is None:
            print("❌ No data loaded. Please check file path.")
            return
        
        print("🚀 Starting Comprehensive EDA Analysis...")
        
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
        
        print(f"\n✅ Analysis Complete!")
        print(f"📁 Dataset: {self.file_path}")
        print(f"📊 Records Analyzed: {len(self.df):,}")

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
