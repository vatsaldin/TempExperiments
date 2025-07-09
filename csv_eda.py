import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CSVExploratoryAnalysis:
    """
    Enhanced EDA class with comprehensive NaN handling for visualizations
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.analysis_folder = "analysis"
        self.text_output = []
        self.chart_counter = 1
        
        # NaN handling configuration
        self.nan_handling_config = {
            'categorical_nan_label': 'Missing/Unknown',
            'show_nan_in_charts': True,
            'nan_color': '#FF6B6B',  # Red color for NaN values
            'min_category_threshold': 0.01,  # Minimum 1% to show category
            'max_categories_display': 15  # Maximum categories to show in charts
        }
        
        self.create_analysis_folder()
        self.text_file_path = os.path.join(self.analysis_folder, "analysis_report.txt")
        self.load_data()
    
    def create_analysis_folder(self):
        """Create analysis folder if it doesn't exist"""
        if not os.path.exists(self.analysis_folder):
            os.makedirs(self.analysis_folder)
            print(f"✅ Created '{self.analysis_folder}' folder")
        else:
            print(f"📁 Using existing '{self.analysis_folder}' folder")
    
    def log_text(self, text):
        """Log text to both console and text file"""
        print(text)
        self.text_output.append(text)
    
    def load_data(self):
        """Load and initial data inspection"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.log_text(f"✅ Data loaded successfully!")
            self.log_text(f"Dataset shape: {self.df.shape}")
            
            # Log initial NaN statistics
            total_nan = self.df.isnull().sum().sum()
            total_cells = self.df.shape[0] * self.df.shape[1]
            nan_percentage = (total_nan / total_cells) * 100
            
            self.log_text(f"📊 NaN Overview: {total_nan:,} missing values ({nan_percentage:.2f}% of total data)")
            
        except Exception as e:
            self.log_text(f"❌ Error loading data: {e}")
            return
    
    def clean_data_for_visualization(self, series, data_type='categorical'):
        """
        Clean data series for visualization by handling NaN values appropriately
        
        Args:
            series: pandas Series to clean
            data_type: 'categorical', 'numerical', or 'text'
            
        Returns:
            cleaned series and metadata about cleaning
        """
        original_length = len(series)
        nan_count = series.isnull().sum()
        
        cleaning_info = {
            'original_length': original_length,
            'nan_count': nan_count,
            'nan_percentage': (nan_count / original_length) * 100 if original_length > 0 else 0,
            'cleaning_method': None
        }
        
        if data_type == 'categorical':
            # For categorical data, replace NaN with descriptive label
            if self.nan_handling_config['show_nan_in_charts'] and nan_count > 0:
                cleaned_series = series.fillna(self.nan_handling_config['categorical_nan_label'])
                cleaning_info['cleaning_method'] = f"NaN values labeled as '{self.nan_handling_config['categorical_nan_label']}'"
            else:
                cleaned_series = series.dropna()
                cleaning_info['cleaning_method'] = "NaN values removed"
                
        elif data_type == 'numerical':
            # For numerical data, typically remove NaN for visualizations
            cleaned_series = series.dropna()
            cleaning_info['cleaning_method'] = "NaN values removed for numerical analysis"
            
        elif data_type == 'text':
            # For text data, remove empty/NaN values
            cleaned_series = series.dropna()
            # Also remove empty strings
            cleaned_series = cleaned_series[cleaned_series.astype(str).str.strip() != '']
            cleaning_info['cleaning_method'] = "NaN and empty values removed"
            
        return cleaned_series, cleaning_info
    
    def prepare_categorical_for_chart(self, series, max_categories=None):
        """
        Prepare categorical data for charting with intelligent NaN handling
        
        Args:
            series: pandas Series with categorical data
            max_categories: Maximum number of categories to display
            
        Returns:
            dict with chart data and metadata
        """
        if max_categories is None:
            max_categories = self.nan_handling_config['max_categories_display']
        
        # Clean the series
        cleaned_series, cleaning_info = self.clean_data_for_visualization(series, 'categorical')
        
        # Get value counts
        value_counts = cleaned_series.value_counts()
        total_count = len(cleaned_series)
        
        # Filter out categories that are too small
        min_count = max(1, int(total_count * self.nan_handling_config['min_category_threshold']))
        significant_categories = value_counts[value_counts >= min_count]
        
        # Handle "Other" category for too many categories
        if len(significant_categories) > max_categories:
            top_categories = significant_categories.head(max_categories - 1)
            other_count = significant_categories.iloc[max_categories-1:].sum()
            
            if other_count > 0:
                top_categories['Other'] = other_count
            
            chart_data = top_categories
        else:
            chart_data = significant_categories
        
        # Prepare colors - highlight NaN category if present
        colors = []
        nan_label = self.nan_handling_config['categorical_nan_label']
        
        for category in chart_data.index:
            if category == nan_label:
                colors.append(self.nan_handling_config['nan_color'])
            else:
                colors.append(None)  # Use default color
        
        return {
            'data': chart_data,
            'colors': colors,
            'cleaning_info': cleaning_info,
            'total_original': len(series),
            'total_cleaned': len(cleaned_series)
        }
    
    def save_chart(self, filename_prefix, title=""):
        """Save current matplotlib figure to analysis folder"""
        filename = f"{self.chart_counter:02d}_{filename_prefix}.png"
        filepath = os.path.join(self.analysis_folder, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        self.log_text(f"📊 Chart saved: {filename}")
        self.chart_counter += 1
        
        return filepath
    
    def data_quality_assessment(self):
        """Enhanced data quality analysis with NaN handling"""
        self.log_text("\n" + "="*60)
        self.log_text("🔍 DATA QUALITY ASSESSMENT")
        self.log_text("="*60)
        
        # Missing values analysis
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Data_Type': self.df.dtypes,
            'Non_Missing_Count': self.df.notnull().sum()
        }).sort_values('Missing_Percentage', ascending=False)
        
        self.log_text("\n📉 Missing Values Summary:")
        self.log_text(missing_data.to_string(index=False))
        
        # Enhanced visualization with NaN handling
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality Overview with NaN Analysis', fontsize=16, fontweight='bold')
        
        # 1. Missing values heatmap (sample)
        sample_size = min(100, len(self.df))
        sample_df = self.df.head(sample_size)
        
        # Create custom colormap for missing values
        from matplotlib.colors import ListedColormap
        colors = ['lightgreen', self.nan_handling_config['nan_color']]
        cmap = ListedColormap(colors)
        
        sns.heatmap(sample_df.isnull(), ax=axes[0,0], cbar=True, 
                   yticklabels=False, cmap=cmap, 
                   xticklabels=True, cbar_kws={'label': 'Missing Values'})
        axes[0,0].set_title(f'Missing Values Pattern (First {sample_size} rows)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Missing values bar plot (only show columns with missing values)
        missing_cols = missing_data[missing_data['Missing_Count'] > 0]
        if len(missing_cols) > 0:
            bars = axes[0,1].bar(range(len(missing_cols)), missing_cols['Missing_Percentage'])
            axes[0,1].set_title('Missing Values by Column (%)')
            axes[0,1].set_xlabel('Columns')
            axes[0,1].set_ylabel('Missing Percentage')
            axes[0,1].set_xticks(range(len(missing_cols)))
            axes[0,1].set_xticklabels(missing_cols['Column'], rotation=45)
            
            # Color bars based on severity
            for i, bar in enumerate(bars):
                pct = missing_cols.iloc[i]['Missing_Percentage']
                if pct > 50:
                    bar.set_color('red')
                elif pct > 20:
                    bar.set_color('orange')
                else:
                    bar.set_color('yellow')
        else:
            axes[0,1].text(0.5, 0.5, 'No Missing Values Found!', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[0,1].transAxes, fontsize=14, color='green')
            axes[0,1].set_title('Missing Values Status')
        
        # 3. Data completeness by column
        completeness = (1 - self.df.isnull().sum() / len(self.df)) * 100
        bars = axes[1,0].bar(range(len(completeness)), completeness.values)
        axes[1,0].set_title('Data Completeness by Column (%)')
        axes[1,0].set_xlabel('Columns')
        axes[1,0].set_ylabel('Completeness Percentage')
        axes[1,0].set_xticks(range(len(completeness)))
        axes[1,0].set_xticklabels(completeness.index, rotation=45)
        axes[1,0].set_ylim(0, 100)
        
        # Color bars based on completeness
        for i, bar in enumerate(bars):
            pct = completeness.iloc[i]
            if pct >= 95:
                bar.set_color('green')
            elif pct >= 80:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 4. Overall data quality summary
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Create a pie chart for overall quality
        sizes = [quality_score, 100 - quality_score]
        labels = ['Complete Data', 'Missing Data']
        colors = ['lightgreen', self.nan_handling_config['nan_color']]
        
        axes[1,1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title(f'Overall Data Quality Score: {quality_score:.1f}%')
        
        plt.tight_layout()
        self.save_chart("data_quality_enhanced", "Enhanced Data Quality with NaN Analysis")
        plt.show()
        
        # Data quality recommendations
        self.log_text(f"\n💡 Data Quality Recommendations:")
        
        high_missing = missing_data[missing_data['Missing_Percentage'] > 50]
        if len(high_missing) > 0:
            self.log_text(f"⚠️  High missing data columns (>50%): {', '.join(high_missing['Column'])}")
            self.log_text(f"   Consider: Drop these columns or investigate data collection process")
        
        medium_missing = missing_data[(missing_data['Missing_Percentage'] > 20) & 
                                     (missing_data['Missing_Percentage'] <= 50)]
        if len(medium_missing) > 0:
            self.log_text(f"⚠️  Medium missing data columns (20-50%): {', '.join(medium_missing['Column'])}")
            self.log_text(f"   Consider: Imputation strategies or special handling")
        
        low_missing = missing_data[(missing_data['Missing_Percentage'] > 0) & 
                                  (missing_data['Missing_Percentage'] <= 20)]
        if len(low_missing) > 0:
            self.log_text(f"✅ Low missing data columns (<20%): {', '.join(low_missing['Column'])}")
            self.log_text(f"   Consider: Simple imputation or case-wise deletion")
        
        return missing_data
    
    def categorical_analysis(self):
        """Enhanced categorical analysis with NaN handling"""
        self.log_text("\n" + "="*60)
        self.log_text("📊 CATEGORICAL DATA ANALYSIS (Enhanced NaN Handling)")
        self.log_text("="*60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            self.log_text("No categorical columns found.")
            return
        
        self.log_text(f"Categorical Columns: {categorical_cols}")
        
        # Analyze each categorical column
        for col in categorical_cols:
            self.log_text(f"\n🏷️ Column: {col}")
            self.log_text("-" * 40)
            
            # Prepare data for visualization
            chart_data = self.prepare_categorical_for_chart(self.df[col])
            
            # Log statistics
            cleaning_info = chart_data['cleaning_info']
            self.log_text(f"Original values: {cleaning_info['original_length']}")
            self.log_text(f"Missing values: {cleaning_info['nan_count']} ({cleaning_info['nan_percentage']:.1f}%)")
            self.log_text(f"Cleaning method: {cleaning_info['cleaning_method']}")
            
            # Display top values
            self.log_text(f"\nTop values (after cleaning):")
            for idx, (value, count) in enumerate(chart_data['data'].items(), 1):
                pct = (count / chart_data['total_cleaned']) * 100
                nan_indicator = " 🔴" if value == self.nan_handling_config['categorical_nan_label'] else ""
                self.log_text(f"{idx:2}. {str(value):<30} | {count:4} ({pct:5.1f}%){nan_indicator}")
        
        # Create enhanced visualizations
        n_cols = len(categorical_cols)
        if n_cols > 0:
            fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(16, 6 * ((n_cols + 1) // 2)))
            fig.suptitle('Categorical Data Distribution (Enhanced NaN Handling)', fontsize=16, fontweight='bold')
            
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
                
                # Prepare chart data
                chart_data = self.prepare_categorical_for_chart(self.df[col])
                
                # Create bar plot with custom colors
                bars = ax.bar(range(len(chart_data['data'])), chart_data['data'].values)
                
                # Apply colors (highlight NaN categories)
                for j, (bar, color) in enumerate(zip(bars, chart_data['colors'])):
                    if color is not None:
                        bar.set_color(color)
                
                ax.set_title(f'{col}\n(Missing: {chart_data["cleaning_info"]["nan_count"]} values)')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                ax.set_xticks(range(len(chart_data['data'])))
                ax.set_xticklabels(chart_data['data'].index, rotation=45, ha='right')
                
                # Add value labels on bars
                for j, (bar, value) in enumerate(zip(bars, chart_data['data'].values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value}', ha='center', va='bottom', fontsize=9)
            
            # Remove empty subplots
            if n_cols % 2 == 1 and n_cols > 1:
                fig.delaxes(axes[-1, -1])
            
            plt.tight_layout()
            self.save_chart("categorical_enhanced", "Enhanced Categorical Analysis with NaN Handling")
            plt.show()
    
    def numerical_analysis(self):
        """Enhanced numerical analysis with NaN handling"""
        self.log_text("\n" + "="*60)
        self.log_text("📈 NUMERICAL DATA ANALYSIS (Enhanced NaN Handling)")
        self.log_text("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            self.log_text("No numerical columns found.")
            return
        
        # Analyze NaN impact on numerical columns
        for col in numerical_cols:
            original_series = self.df[col]
            cleaned_series, cleaning_info = self.clean_data_for_visualization(original_series, 'numerical')
            
            self.log_text(f"\n📊 Column: {col}")
            self.log_text(f"   Original values: {cleaning_info['original_length']}")
            self.log_text(f"   Missing values: {cleaning_info['nan_count']} ({cleaning_info['nan_percentage']:.1f}%)")
            self.log_text(f"   Available for analysis: {len(cleaned_series)}")
            
            if len(cleaned_series) > 0:
                self.log_text(f"   Mean: {cleaned_series.mean():.2f}")
                self.log_text(f"   Std: {cleaned_series.std():.2f}")
                self.log_text(f"   Min: {cleaned_series.min():.2f}")
                self.log_text(f"   Max: {cleaned_series.max():.2f}")
        
        # Create enhanced visualizations
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots((n_cols + 2) // 3, 3, figsize=(18, 6 * ((n_cols + 2) // 3)))
        fig.suptitle('Numerical Data Distribution (NaN Values Excluded)', fontsize=16, fontweight='bold')
        
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
            
            # Clean data for visualization
            cleaned_series, cleaning_info = self.clean_data_for_visualization(self.df[col], 'numerical')
            
            if len(cleaned_series) > 0:
                # Create histogram
                ax.hist(cleaned_series, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
                
                # Add statistics lines
                mean_val = cleaned_series.mean()
                median_val = cleaned_series.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, 
                          label=f'Median: {median_val:.2f}')
                
                # Title with missing value info
                title = f'{col}\n(Missing: {cleaning_info["nan_count"]} values, {cleaning_info["nan_percentage"]:.1f}%)'
                ax.set_title(title)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.legend()
                
                # Add text box with statistics
                stats_text = f'Available: {len(cleaned_series)}\nStd: {cleaned_series.std():.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'No data available\n(All values are NaN)', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_title(f'{col} - No Data Available')
        
        # Remove empty subplots
        for i in range(n_cols, len(axes.flat)):
            fig.delaxes(axes.flat[i])
        
        plt.tight_layout()
        self.save_chart("numerical_enhanced", "Enhanced Numerical Analysis with NaN Handling")
        plt.show()
        
        # Correlation analysis with NaN handling
        if len(numerical_cols) > 1:
            self.log_text("\n🔗 Correlation Analysis (NaN values excluded):")
            
            # Create correlation matrix excluding NaN values
            clean_numeric_df = self.df[numerical_cols].dropna()
            
            if len(clean_numeric_df) > 0:
                correlation_matrix = clean_numeric_df.corr()
                
                plt.figure(figsize=(10, 8))
                
                # Create mask for better visualization
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                # Generate heatmap
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, mask=mask, fmt='.3f')
                
                plt.title(f'Correlation Matrix (Based on {len(clean_numeric_df)} complete cases)')
                plt.tight_layout()
                self.save_chart("correlation_enhanced", "Enhanced Correlation Analysis")
                plt.show()
                
                # Log missing data impact on correlation
                original_length = len(self.df)
                available_for_corr = len(clean_numeric_df)
                excluded_pct = ((original_length - available_for_corr) / original_length) * 100
                
                self.log_text(f"   Original rows: {original_length}")
                self.log_text(f"   Complete cases: {available_for_corr}")
                self.log_text(f"   Excluded due to NaN: {original_length - available_for_corr} ({excluded_pct:.1f}%)")
            else:
                self.log_text("   ⚠️ No complete cases available for correlation analysis")
    
    def response_analysis(self):
        """Enhanced response analysis with NaN handling"""
        if 'RESPONSE' in self.df.columns:
            self.log_text("\n" + "="*60)
            self.log_text("✅ RESPONSE ANALYSIS (Enhanced NaN Handling)")
            self.log_text("="*60)
            
            # Prepare response data
            chart_data = self.prepare_categorical_for_chart(self.df['RESPONSE'])
            
            self.log_text("Response Distribution (after NaN handling):")
            for response, count in chart_data['data'].items():
                pct = (count / chart_data['total_cleaned']) * 100
                nan_indicator = " 🔴" if response == self.nan_handling_config['categorical_nan_label'] else ""
                self.log_text(f"{response:<20}: {count:4} ({pct:5.1f}%){nan_indicator}")
            
            # Enhanced visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Response Analysis with NaN Handling', fontsize=16, fontweight='bold')
            
            # 1. Response distribution pie chart
            colors = ['lightblue', 'lightgreen', 'lightcoral', self.nan_handling_config['nan_color']]
            colors = colors[:len(chart_data['data'])]
            
            axes[0,0].pie(chart_data['data'].values, labels=chart_data['data'].index, 
                         autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0,0].set_title('Response Distribution')
            
            # 2. Response bar chart
            bars = axes[0,1].bar(range(len(chart_data['data'])), chart_data['data'].values)
            axes[0,1].set_title('Response Count')
            axes[0,1].set_xlabel('Response Type')
            axes[0,1].set_ylabel('Count')
            axes[0,1].set_xticks(range(len(chart_data['data'])))
            axes[0,1].set_xticklabels(chart_data['data'].index, rotation=45)
            
            # Color bars to highlight NaN
            for i, (bar, color) in enumerate(zip(bars, chart_data['colors'])):
                if color is not None:
                    bar.set_color(color)
            
            # 3. Response by other categorical variables (if available)
            categorical_cols = ['SITE_NAME', 'FACILITY_AREA', 'AUTHOR_NAME']
            available_cols = [col for col in categorical_cols if col in self.df.columns]
            
            if available_cols:
                # Create cross-tabulation with NaN handling
                for i, col in enumerate(available_cols[:2]):
                    ax = axes[1, i]
                    
                    # Prepare both columns for cross-tabulation
                    response_clean = self.df['RESPONSE'].fillna(self.nan_handling_config['categorical_nan_label'])
                    col_clean = self.df[col].fillna(self.nan_handling_config['categorical_nan_label'])
                    
                    # Create cross-tabulation
                    crosstab = pd.crosstab(col_clean, response_clean)
                    
                    # Limit categories for readability
                    if len(crosstab) > 10:
                        crosstab = crosstab.head(10)
                    
                    crosstab.plot(kind='bar', ax=ax, stacked=True)
                    ax.set_title(f'Response by {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            self.save_chart("response_enhanced", "Enhanced Response Analysis with NaN Handling")
            plt.show()
            
            # Log NaN impact
            cleaning_info = chart_data['cleaning_info']
            self.log_text(f"\n📊 NaN Impact on Response Analysis:")
            self.log_text(f"   Original responses: {cleaning_info['original_length']}")
            self.log_text(f"   Missing responses: {cleaning_info['nan_count']} ({cleaning_info['nan_percentage']:.1f}%)")
            self.log_text(f"   Cleaning method: {cleaning_info['cleaning_method']}")
    
    def save_text_output(self):
        """Save all text output to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{'='*80}
ENHANCED CSV EXPLORATORY DATA ANALYSIS REPORT
(With Advanced NaN Handling for Visualizations)
Generated on: {timestamp}
Dataset: {self.file_path}
{'='*80}

NaN HANDLING CONFIGURATION:
- Categorical NaN Label: '{self.nan_handling_config['categorical_nan_label']}'
- Show NaN in Charts: {self.nan_handling_config['show_nan_in_charts']}
- NaN Color: {self.nan_handling_config['nan_color']}
- Min Category Threshold: {self.nan_handling_config['min_category_threshold']*100}%
- Max Categories Display: {self.nan_handling_config['max_categories_display']}

"""
        
        with open(self.text_file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write('\n'.join(self.text_output))
        
        print(f"\n💾 Enhanced text analysis saved to: {self.text_file_path}")
    
    def run_enhanced_analysis(self):
        """Run the enhanced EDA pipeline with comprehensive NaN handling"""
        if self.df is None:
            self.log_text("❌ No data loaded. Please check file path.")
            return
        
        self.log_text("🚀 Starting Enhanced EDA Analysis with NaN Handling...")
        
        # Run all analysis steps
        self.data_quality_assessment()
        self.categorical_analysis()
        self.numerical_analysis()
        self.response_analysis()
        
        # Save all outputs
        self.save_text_output()
        
        self.log_text(f"\n✅ Enhanced Analysis Complete!")
        self.log_text(f"📁 Dataset: {self.file_path}")
        self.log_text(f"📊 Records Analyzed: {len(self.df):,}")
        self.log_text(f"📈 Charts Generated: {self.chart_counter - 1}")
        self.log_text(f"📄 Report saved to: {self.text_file_path}")
        self.log_text(f"🎯 All outputs saved to: {self.analysis_folder}/")

# Additional utility functions for specific NaN handling scenarios

def create_nan_summary_report(df, output_file="nan_summary_report.txt"):
    """
    Create a comprehensive NaN summary report
    
    Args:
        df: pandas DataFrame
        output_file: Output file path for the report
    """
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE NaN ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total_cells = df.shape[0] * df.shape[1]
        total_nan = df.isnull().sum().sum()
        nan_percentage = (total_nan / total_cells) * 100
        
        f.write(f"OVERALL STATISTICS:\n")
        f.write(f"Total cells: {total_cells:,}\n")
        f.write(f"Total NaN values: {total_nan:,}\n")
        f.write(f"NaN percentage: {nan_percentage:.2f}%\n\n")
        
        # Per-column analysis
        f.write("PER-COLUMN NaN ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        
        for col in df.columns:
            col_nan = df[col].isnull().sum()
            col_total = len(df[col])
            col_nan_pct = (col_nan / col_total) * 100
            
            f.write(f"\n{col}:\n")
            f.write(f"  Data type: {df[col].dtype}\n")
            f.write(f"  Total values: {col_total:,}\n")
            f.write(f"  NaN values: {col_nan:,}\n")
            f.write(f"  NaN percentage: {col_nan_pct:.2f}%\n")
            
            if col_nan > 0:
                # NaN patterns
                nan_indices = df[df[col].isnull()].index.tolist()
                f.write(f"  First few NaN indices: {nan_indices[:10]}\n")
                
                # Check for patterns
                if len(nan_indices) > 1:
                    consecutive_groups = []
                    current_group = [nan_indices[0]]
                    
                    for i in range(1, len(nan_indices)):
                        if nan_indices[i] == nan_indices[i-1] + 1:
                            current_group.append(nan_indices[i])
                        else:
                            if len(current_group) > 1:
                                consecutive_groups.append(current_group)
                            current_group = [nan_indices[i]]
                    
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    
                    if consecutive_groups:
                        f.write(f"  Consecutive NaN groups found: {len(consecutive_groups)}\n")
                        for j, group in enumerate(consecutive_groups[:3]):
                            f.write(f"    Group {j+1}: indices {group[0]}-{group[-1]} (length: {len(group)})\n")
        
        # Correlation between missing values
        f.write("\n" + "="*50 + "\n")
        f.write("MISSING VALUE CORRELATIONS:\n")
        f.write("="*50 + "\n")
        
        # Create missing value correlation matrix
        missing_df = df.isnull().astype(int)
        missing_corr = missing_df.corr()
        
        # Find highly correlated missing patterns
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_corr_pairs.append((missing_corr.columns[i], missing_corr.columns[j], corr_val))
        
        if high_corr_pairs:
            f.write("Highly correlated missing patterns (|correlation| > 0.5):\n")
            for col1, col2, corr in high_corr_pairs:
                f.write(f"  {col1} <-> {col2}: {corr:.3f}\n")
        else:
            f.write("No highly correlated missing patterns found.\n")
        
        # Recommendations
        f.write("\n" + "="*50 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*50 + "\n")
        
        # Column-specific recommendations
        for col in df.columns:
            col_nan_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if col_nan_pct > 70:
                f.write(f"\n{col}: DROP COLUMN (>70% missing)\n")
                f.write(f"  - Consider removing this column from analysis\n")
                f.write(f"  - Investigate data collection process\n")
                
            elif col_nan_pct > 30:
                f.write(f"\n{col}: SPECIAL HANDLING REQUIRED (>30% missing)\n")
                f.write(f"  - Consider advanced imputation techniques\n")
                f.write(f"  - Create missing value indicator variable\n")
                f.write(f"  - Investigate systematic missing patterns\n")
                
            elif col_nan_pct > 5:
                f.write(f"\n{col}: MODERATE MISSING DATA (>5% missing)\n")
                if df[col].dtype in ['int64', 'float64']:
                    f.write(f"  - Consider median/mean imputation\n")
                    f.write(f"  - Try regression imputation\n")
                else:
                    f.write(f"  - Consider mode imputation\n")
                    f.write(f"  - Create 'Unknown' category\n")
                    
            elif col_nan_pct > 0:
                f.write(f"\n{col}: LOW MISSING DATA (<5% missing)\n")
                f.write(f"  - Simple imputation or listwise deletion acceptable\n")
    
    print(f"📄 Comprehensive NaN report saved to: {output_file}")

def handle_categorical_nan_for_modeling(df, columns, strategy='mode'):
    """
    Handle NaN values in categorical columns for modeling
    
    Args:
        df: pandas DataFrame
        columns: list of categorical column names
        strategy: 'mode', 'unknown', or 'frequent'
    
    Returns:
        DataFrame with handled NaN values
    """
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            if strategy == 'mode':
                # Fill with most frequent value
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_value[0])
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    
            elif strategy == 'unknown':
                # Fill with 'Unknown' category
                df_clean[col] = df_clean[col].fillna('Unknown')
                
            elif strategy == 'frequent':
                # Fill with most frequent non-null value
                value_counts = df_clean[col].value_counts()
                if len(value_counts) > 0:
                    df_clean[col] = df_clean[col].fillna(value_counts.index[0])
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
    
    return df_clean

def handle_numerical_nan_for_modeling(df, columns, strategy='median'):
    """
    Handle NaN values in numerical columns for modeling
    
    Args:
        df: pandas DataFrame
        columns: list of numerical column names
        strategy: 'median', 'mean', 'mode', or 'interpolate'
    
    Returns:
        DataFrame with handled NaN values
    """
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            if strategy == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
            elif strategy == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                
            elif strategy == 'mode':
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_value[0])
                else:
                    df_clean[col] = df_clean[col].fillna(0)
                    
            elif strategy == 'interpolate':
                df_clean[col] = df_clean[col].interpolate()
                # Fill any remaining NaN with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

def create_missing_value_indicators(df, columns, threshold=0.05):
    """
    Create indicator variables for missing values
    
    Args:
        df: pandas DataFrame
        columns: list of column names to create indicators for
        threshold: minimum missing percentage to create indicator
    
    Returns:
        DataFrame with missing value indicators
    """
    
    df_with_indicators = df.copy()
    
    for col in columns:
        if col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct >= threshold:
                indicator_col = f"{col}_missing"
                df_with_indicators[indicator_col] = df[col].isnull().astype(int)
                print(f"Created missing indicator for {col} (missing: {missing_pct:.1%})")
    
    return df_with_indicators

# Example usage functions

def run_enhanced_eda_with_nan_handling(file_path):
    """
    Run enhanced EDA with comprehensive NaN handling
    
    Args:
        file_path: path to CSV file
    """
    
    print("🚀 Starting Enhanced EDA with NaN Handling...")
    print("="*60)
    
    # Initialize analyzer
    analyzer = CSVExploratoryAnalysis(file_path)
    
    # Run enhanced analysis
    analyzer.run_enhanced_analysis()
    
    # Create additional NaN summary report
    print("\n📋 Creating comprehensive NaN summary report...")
    create_nan_summary_report(analyzer.df, 
                             os.path.join(analyzer.analysis_folder, "nan_summary_report.txt"))
    
    print(f"\n✅ Enhanced EDA with NaN handling complete!")
    print(f"📁 All outputs saved to: {analyzer.analysis_folder}/")
    
    return analyzer

def prepare_data_for_modeling(df, target_column=None):
    """
    Prepare data for modeling with intelligent NaN handling
    
    Args:
        df: pandas DataFrame
        target_column: name of target column (if any)
    
    Returns:
        dict with cleaned data and metadata
    """
    
    print("🔧 Preparing data for modeling...")
    
    # Separate features and target
    if target_column and target_column in df.columns:
        features = df.drop(columns=[target_column])
        target = df[target_column]
    else:
        features = df.copy()
        target = None
    
    # Identify column types
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"📊 Data overview:")
    print(f"   Total features: {len(features.columns)}")
    print(f"   Categorical: {len(categorical_cols)}")
    print(f"   Numerical: {len(numerical_cols)}")
    
    # Handle categorical NaN
    if categorical_cols:
        print(f"\n🏷️ Handling categorical NaN values...")
        features_clean = handle_categorical_nan_for_modeling(features, categorical_cols, strategy='unknown')
        
        for col in categorical_cols:
            original_nan = features[col].isnull().sum()
            if original_nan > 0:
                print(f"   {col}: {original_nan} NaN values → filled with 'Unknown'")
    else:
        features_clean = features.copy()
    
    # Handle numerical NaN
    if numerical_cols:
        print(f"\n📈 Handling numerical NaN values...")
        features_clean = handle_numerical_nan_for_modeling(features_clean, numerical_cols, strategy='median')
        
        for col in numerical_cols:
            original_nan = features[col].isnull().sum()
            if original_nan > 0:
                print(f"   {col}: {original_nan} NaN values → filled with median")
    
    # Create missing value indicators
    print(f"\n🔍 Creating missing value indicators...")
    features_with_indicators = create_missing_value_indicators(features_clean, 
                                                             features.columns.tolist(), 
                                                             threshold=0.05)
    
    # Final validation
    remaining_nan = features_with_indicators.isnull().sum().sum()
    print(f"\n✅ Data preparation complete!")
    print(f"   Remaining NaN values: {remaining_nan}")
    print(f"   Final feature count: {len(features_with_indicators.columns)}")
    
    return {
        'features': features_with_indicators,
        'target': target,
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_cols,
        'original_nan_count': df.isnull().sum().sum(),
        'final_nan_count': remaining_nan
    }

# Example usage
if __name__ == "__main__":
    # Example 1: Run enhanced EDA with NaN handling
    print("="*80)
    print("ENHANCED CSV EDA WITH NaN HANDLING - EXAMPLE USAGE")
    print("="*80)
    
    # Replace with your CSV file path
    file_path = "your_data.csv"
    
    try:
        # Run enhanced analysis
        analyzer = run_enhanced_eda_with_nan_handling(file_path)
        
        # Example 2: Prepare data for modeling
        print("\n" + "="*60)
        print("PREPARING DATA FOR MODELING")
        print("="*60)
        
        modeling_data = prepare_data_for_modeling(analyzer.df, target_column='RESPONSE')
        
        print(f"\n🎯 Data ready for modeling:")
        print(f"   Features shape: {modeling_data['features'].shape}")
        if modeling_data['target'] is not None:
            print(f"   Target shape: {modeling_data['target'].shape}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please update the file_path variable with your actual CSV file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("KEY FEATURES OF ENHANCED NaN HANDLING:")
    print("="*80)
    print("✅ Intelligent NaN detection and labeling in charts")
    print("✅ Custom colors for missing value categories")
    print("✅ Comprehensive missing value impact analysis")
    print("✅ Smart data cleaning for different data types")
    print("✅ Missing value correlation analysis")
    print("✅ Automated recommendations for NaN handling")
    print("✅ Model-ready data preparation")
    print("✅ Detailed reporting and visualization")
    print("✅ Configurable NaN handling strategies")
    print("✅ Professional charts without 'nan' labels")
    print("\n🚀 Ready to use with your data!")
