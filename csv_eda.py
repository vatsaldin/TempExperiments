def response_analysis(self):
        """Enhanced response analysis with NaN handling and separate subplot saving"""
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
            
            # Enhanced visualization with separate subplot saving
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
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, chart_data['data'].values)):
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value}', ha='center', va='bottom', fontsize=10)
            
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
            
            # Save the combined dashboard
            self.save_chart("response_dashboard", "Enhanced Response Analysis with NaN Handling", fig)
            
            # Save individual subplots separately
            subplot_titles = [
                'Response Distribution Pie Chart',
                'Response Count Bar Chart',
                f'Response by {available_cols[0]}' if available_cols else 'Response Analysis 3',
                f'Response by {available_cols[1]}' if len(available_cols) > 1 else 'Response Analysis 4'
            ]
            
            self.log_text("\n📊 Saving individual response plots...")
            saved_files = self.save_subplot_separately(fig, axes, "response", subplot_titles)
            
            for i, filepath in enumerate(saved_files):
                self.log_text(f"   📋 Response plot {i+1} saved: {os.path.basename(filepath)}")
            
            plt.show()
            
            # Create detailed response analysis
            self.create_detailed_response_analysis(chart_data, available_cols)
            
            # Log NaN impact
            cleaning_info = chart_data['cleaning_info']
            self.log_text(f"\n📊 NaN Impact on Response Analysis:")
            self.log_text(f"   Original responses: {cleaning_info['original_length']}")
            self.log_text(f"   Missing responses: {cleaning_info['nan_count']} ({cleaning_info['nan_percentage']:.1f}%)")
            self.log_text(f"   Cleaning method: {cleaning_info['cleaning_method']}")
    
    def create_detailed_response_analysis(self, chart_data, available_cols):
        """Create detailed response analysis with cross-tabulations"""
        self.log_text("\n📊 Creating detailed response analysis...")
        
        # Create detailed cross-tabulation analysis
        if available_cols:
            for col in available_cols:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Detailed Response Analysis by {col}', fontsize=16, fontweight='bold')
                
                # Prepare data
                response_clean = self.df['RESPONSE'].fillna(self.nan_handling_config['categorical_nan_label'])
                col_clean = self.df[col].fillna(self.nan_handling_config['categorical_nan_label'])
                
                # Create cross-tabulation
                crosstab = pd.crosstab(col_clean, response_clean)
                crosstab_pct = pd.crosstab(col_clean, response_clean, normalize='index') * 100
                
                # Plot 1: Stacked bar chart (counts)
                crosstab.plot(kind='bar', ax=ax1, stacked=True)
                ax1.set_title(f'Response Counts by {col}')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                ax1.legend(title='Response')
                
                # Plot 2: Stacked bar chart (percentages)
                crosstab_pct.plot(kind='bar', ax=ax2, stacked=True)
                ax2.set_title(f'Response Percentages by {col}')
                ax2.set_xlabel(col)
                ax2.set_ylabel('Percentage')
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend(title='Response')
                
                # Plot 3: Grouped bar chart
                crosstab.plot(kind='bar', ax=ax3, stacked=False)
                ax3.set_title(f'Response Distribution by {col} (Grouped)')
                ax3.set_xlabel(col)
                ax3.set_ylabel('Count')
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend(title='Response')
                
                # Plot 4: Summary table
                ax4.axis('off')
                
                # Create summary statistics
                summary_text = f"Cross-tabulation Summary: {col} vs RESPONSE\n\n"
                summary_text += f"Total combinations: {len(crosstab) * len(crosstab.columns)}\n"
                summary_text += f"Categories in {col}: {len(crosstab)}\n"
                summary_text += f"Response types: {len(crosstab.columns)}\n\n"
                
                # Add top combinations
                summary_text += "Top 5 combinations:\n"
                flat_crosstab = crosstab.stack().sort_values(ascending=False)
                for i, (idx, count) in enumerate(flat_crosstab.head(5).items()):
                    summary_text += f"{i+1}. {idx[0]} + {idx[1]}: {count}\n"
                
                ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                plt.tight_layout()
                
                # Save detailed response analysis
                detailed_filename = f"response_detailed_{col.replace(' ', '_').replace('/', '_')}"
                self.save_chart(detailed_filename, f"Detailed Response Analysis by {col}", fig)
                
                plt.show()
                plt.close(fig)
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard with all key metrics"""
        self.log_text("\n📊 Creating comprehensive summary dashboard...")
        
        # Create a large summary figure
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('Comprehensive EDA Summary Dashboard', fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        # 1. Data Overview (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        overview_text = f"""
        DATASET OVERVIEW
        
        Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
        
        Missing Values: {self.df.isnull().sum().sum():,} total
        Data Quality: {((1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100):.1f}%
        
        Column Types:
        • Categorical: {len(self.df.select_dtypes(include=['object']).columns)}
        • Numerical: {len(self.df.select_dtypes(include=[np.number]).columns)}
        """
        ax1.text(0.05, 0.95, overview_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax1.axis('off')
        
        # 2. Missing Values Summary
        ax2 = fig.add_subplot(gs[0, 2:])
        missing_summary = self.df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
        
        if len(missing_summary) > 0:
            bars = ax2.bar(range(len(missing_summary)), missing_summary.values, color='red', alpha=0.7)
            ax2.set_title('Missing Values by Column')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('Missing Count')
            ax2.set_xticks(range(len(missing_summary)))
            ax2.set_xticklabels(missing_summary.index, rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, missing_summary.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14, color='green', weight='bold')
            ax2.set_title('Missing Values Status')
        
        # 3. Categorical columns summary (if any)
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            for i, col in enumerate(categorical_cols[:4]):  # Show first 4 categorical columns
                ax = fig.add_subplot(gs[1 + i//2, (i%2)*2:(i%2)*2+2])
                
                chart_data = self.prepare_categorical_for_chart(self.df[col], max_categories=10)
                
                bars = ax.bar(range(len(chart_data['data'])), chart_data['data'].values)
                
                # Apply colors
                for j, (bar, color) in enumerate(zip(bars, chart_data['colors'])):
                    if color is not None:
                        bar.set_color(color)
                
                ax.set_title(f'{col} (Top 10)')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                ax.set_xticks(range(len(chart_data['data'])))
                ax.set_xticklabels(chart_data['data'].index, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, chart_data['data'].values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value}', ha='center', va='bottom', fontsize=8)
        
        # 4. Numerical columns summary (if any)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            for i, col in enumerate(numerical_cols[:4]):  # Show first 4 numerical columns
                ax = fig.add_subplot(gs[3 + i//2, (i%2)*2:(i%2)*2+2])
                
                cleaned_series, cleaning_info = self.clean_data_for_visualization(self.df[col], 'numerical')
                
                if len(cleaned_series) > 0:
                    ax.hist(cleaned_series, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
                    
                    mean_val = cleaned_series.mean()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    
                    ax.set_title(f'{col} (Missing: {cleaning_info["nan_count"]})')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No Data\n(All NaN)', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, color='red')
                    ax.set_title(f'{col} - No Data')
        
        # 5. Response Analysis (if RESPONSE column exists)
        if 'RESPONSE' in self.df.columns:
            ax_response = fig.add_subplot(gs[5, :2])
            
            chart_data = self.prepare_categorical_for_chart(self.df['RESPONSE'])
            
            colors = ['lightblue', 'lightgreen', 'lightcoral', self.nan_handling_config['nan_color']]
            colors = colors[:len(chart_data['data'])]
            
            wedges, texts, autotexts = ax_response.pie(chart_data['data'].values, 
                                                      labels=chart_data['data'].index,
                                                      autopct='%1.1f%%', 
                                                      colors=colors,
                                                      startangle=90)
            ax_response.set_title('Response Distribution')
            
            # Adjust text size
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                autotext.set_weight('bold')
        
        # 6. Data Quality Score
        ax_quality = fig.add_subplot(gs[5, 2:])
        
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Create quality gauge
        sizes = [quality_score, 100 - quality_score]
        labels = ['Complete', 'Missing']
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax_quality.pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
        ax_quality.set_title(f'Overall Data Quality: {quality_score:.1f}%')
        
        # Adjust text size
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        # Save summary dashboard
        self.save_chart("summary_dashboard", "Comprehensive EDA Summary Dashboard", fig)
        
        plt.show()
        plt.close(fig)
    
    def run_enhanced_analysis(self):
        """Run the enhanced EDA pipeline with comprehensive NaN handling and separate subplot saving"""
        if self.df is None:
            self.log_text("❌ No data loaded. Please check file path.")
            return
        
        self.log_text("🚀 Starting Enhanced EDA Analysis with NaN Handling and Separate Subplot Saving...")
        
        # Run all analysis steps
        self.data_quality_assessment()
        self.categorical_analysis()
        self.numerical_analysis()
        self.response_analysis()
        
        # Create comprehensive summary dashboard
        self.create_summary_dashboard()
        
        # Save all outputs
        self.save_text_output()
        
        self.log_text(f"\n✅ Enhanced Analysis Complete!")
        self.log_text(f"📁 Dataset: {self.file_path}")
        self.log_text(f"📊 Records Analyzed: {len(self.df):,}")
        self.log_text(f"📈 Charts Generated: {self.chart_counter - 1}")
        self.log_text(f"📄 Report saved to: {self.text_file_path}")
        self.log_text(f"🎯 All outputs saved to: {self.analysis_folder}/")
        
        # Create file summary
        self.create_file_summary()
    
    def create_file_summary(self):
        """Create a summary file listing all generated files with descriptions"""
        summary_path = os.path.join(self.analysis_folder, "file_summary_with_subplots.txt")
        
        # Get list of all files in analysis folder
        analysis_files = os.listdir(self.analysis_folder)
        chart_files = [f for f in analysis_files if f.endswith('.png')]
        
        # Categorize files
        dashboard_files = [f for f in chart_files if 'dashboard' in f and 'subplot' not in f]
        subplot_files = [f for f in chart_files if 'subplot' in f]
        detailed_files = [f for f in chart_files if 'detailed' in f]
        other_files = [f for f in chart_files if f not in dashboard_files + subplot_files + detailed_files]
        
        summary_content = f"""
{'='*80}
EDA ANALYSIS FILES SUMMARY (Enhanced with Separate Subplots)
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {self.file_path}
{'='*80}

📊 TOTAL FILES GENERATED: {len(chart_files)} charts + {len(analysis_files) - len(chart_files)} reports

📋 DASHBOARD FILES ({len(dashboard_files)} files):
{'='*50}
"""
        
        for i, file in enumerate(sorted(dashboard_files), 1):
            summary_content += f"{i:2}. {file}\n"
            summary_content += f"    - Combined view of multiple visualizations\n"
        
        summary_content += f"""
🔍 INDIVIDUAL SUBPLOT FILES ({len(subplot_files)} files):
{'='*50}
"""
        
        # Group subplot files by category
        categories = {}
        for file in subplot_files:
            if 'data_quality' in file:
                categories.setdefault('Data Quality', []).append(file)
            elif 'categorical' in file:
                categories.setdefault('Categorical Analysis', []).append(file)
            elif 'numerical' in file:
                categories.setdefault('Numerical Analysis', []).append(file)
            elif 'response' in file:
                categories.setdefault('Response Analysis', []).append(file)
            elif 'correlation' in file:
                categories.setdefault('Correlation Analysis', []).append(file)
            else:
                categories.setdefault('Other', []).append(file)
        
        for category, files in categories.items():
            summary_content += f"\n{category}:\n"
            for file in sorted(files):
                summary_content += f"  • {file}\n"
        
        summary_content += f"""
📈 DETAILED ANALYSIS FILES ({len(detailed_files)} files):
{'='*50}
"""
        
        for i, file in enumerate(sorted(detailed_files), 1):
            summary_content += f"{i:2}. {file}\n"
            summary_content += f"    - In-depth analysis with multiple views\n"
        
        if other_files:
            summary_content += f"""
📊 OTHER CHART FILES ({len(other_files)} files):
{'='*50}
"""
            for i, file in enumerate(sorted(other_files), 1):
                summary_content += f"{i:2}. {file}\n"
        
        summary_content += f"""
📄 TEXT REPORTS:
{'='*50}
1. analysis_report.txt - Complete text analysis
2. file_summary_with_subplots.txt - This file summary

📁 FOLDER STRUCTURE:
{'='*50}
analysis/
├── analysis_report.txt                    (Complete text analysis)
├── file_summary_with_subplots.txt        (This summary)
├── [Dashboard Files]                      (Combined visualizations)
├── [Subplot Files]                       (Individual subplot extractions)
├── [Detailed Files]                      (In-depth individual analyses)
└── [Other Charts]                        (Additional visualizations)

🔧 USAGE RECOMMENDATIONS:
{'='*50}
1. START WITH DASHBOARDS: Review combined visualizations first
2. EXAMINE SUBPLOTS: Look at individual charts for detailed analysis
3. DIVE INTO DETAILED FILES: For comprehensive single-column analysis
4. READ TEXT REPORT: For statistical summaries and insights
5. SHARE SELECTIVELY: Use individual subplot files for presentations

💡 BENEFITS OF SEPARATE SUBPLOTS:
{'='*50}
✅ Better visibility - Each chart in full resolution
✅ Easier sharing - Individual files for specific insights
✅ Focused analysis - Concentrate on one aspect at a time
✅ Presentation ready - Perfect for reports and slides
✅ Reduced clutter - No overlapping elements
✅ Custom sizing - Each chart optimized for its content

📊 ANALYSIS SECTIONS COVERED:
{'='*50}
1. Data Quality Assessment (Missing values, completeness)
2. Categorical Data Analysis (Distribution, patterns)
3. Numerical Data Analysis (Statistics, distributions)
4. Response Pattern Analysis (If RESPONSE column exists)
5. Correlation Analysis (Numerical relationships)
6. Comprehensive Summary Dashboard (Overall view)

🎯 NEXT STEPS:
{'='*50}
• Use individual subplot files for detailed examination
• Include relevant charts in your reports and presentations
• Refer to text analysis for statistical insights
• Consider the recommendations provided in the analysis report
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📋 Enhanced file summary saved to: {summary_path}")
        print(f"📊 Total charts generated: {len(chart_files)}")
        print(f"   - Dashboard files: {len(dashboard_files)}")
        print(f"   - Individual subplot files: {len(subplot_files)}")
        print(f"   - Detailed analysis files: {len(detailed_files)}")
        print(f"   - Other chart files: {len(other_files)}")

# Example usage function with separate subplot saving
def run_enhanced_eda_with_separate_subplots(file_path):
    """
    Run enhanced EDA with comprehensive NaN handling and separate subplot saving
    
    Args:
        file_path: path to CSV file
    """
    
    print("🚀 Starting Enhanced EDA with NaN Handling and Separate Subplot Saving...")
    print("="*80)
    
    # Initialize analyzer
    analyzer = CSVExploratoryAnalysis(file_path)
    
    # Run enhanced analysis with separate subplot saving
    analyzer.run_enhanced_analysis()
    
    # Create additional NaN summary report
    print("\n📋 Creating comprehensive NaN summary report...")
    create_nan_summary_report(analyzer.df, 
                             os.path.join(analyzer.analysis_folder, "nan_summary_report.txt"))
    
    print(f"\n✅ Enhanced EDA with separate subplot saving complete!")
    print(f"📁 All outputs saved to: {analyzer.analysis_folder}/")
    print(f"📊 Individual subplot files created for better visibility")
    print(f"📋 Check file_summary_with_subplots.txt for detailed file listing")
    
    return analyzer

# Update the main example usage
if __name__ == "__main__":
    # Example 1: Run enhanced EDA with separate subplot saving
    print("="*80)
    print("ENHANCED CSV EDA WITH NaN HANDLING AND SEPARATE SUBPLOTS")
    print("="*80)
    
    # Replace with your CSV file path
    file_path = "your_data.csv"
    
    try:
        # Run enhanced analysis with separate subplot saving
        analyzer = run_enhanced_eda_with_separate_subplots(file_path)
        
        print(f"\n🎯 Analysis Complete!")
        print(f"📊 Original data shape: {analyzer.df.shape}")
        print(f"📈 Charts generated: {analyzer.chart_counter - 1}")
        print(f"📁 Check '{analyzer.analysis_folder}' folder for all outputs")
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please update the file_path variable with your actual CSV file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("KEY FEATURES OF ENHANCED NaN HANDLING WITH SEPARATE SUBPLOTS:")
    print("="*80)
    print("✅ Individual subplot files for better visibility")
    print("✅ Dashboard files for combined overview")
    print("✅ Detailed analysis files for in-depth examination")
    print("✅ Intelligent NaN detection and labeling")
    print("✅ Custom colors for missing value categories")
    print("✅ Comprehensive missing value impact analysis")
    print("✅ Smart data cleaning for different data types")
    print("✅ Professional charts without 'nan' labels")
    print("✅ Separate correlation analysis with multiple views")
    print("✅ Response analysis with cross-tabulations")
    print("✅ File categorization and comprehensive summaries")
    print("✅ Presentation-ready individual chart files")
    print("\n🚀 Perfect for detailed analysis and professional reporting!")
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
    
    def save_chart(self, filename_prefix, title="", fig=None):
        """Save current matplotlib figure to analysis folder"""
        if fig is None:
            fig = plt.gcf()
        
        filename = f"{self.chart_counter:02d}_{filename_prefix}.png"
        filepath = os.path.join(self.analysis_folder, filename)
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        self.log_text(f"📊 Chart saved: {filename}")
        self.chart_counter += 1
        
        return filepath
    
    def save_subplot_separately(self, fig, axes, filename_prefix, subplot_titles):
        """
        Save each subplot as a separate file
        
        Args:
            fig: matplotlib figure object
            axes: axes array from subplots
            filename_prefix: prefix for filenames
            subplot_titles: list of titles for each subplot
        """
        saved_files = []
        
        # Handle different axes structures
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        elif axes.ndim == 2:
            axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < len(subplot_titles):
                # Create new figure for this subplot
                new_fig, new_ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Copy the subplot content
                self.copy_subplot_content(ax, new_ax)
                
                # Set title
                new_ax.set_title(subplot_titles[i], fontsize=14, fontweight='bold')
                
                # Save individual subplot
                subplot_filename = f"{filename_prefix}_subplot_{i+1}_{subplot_titles[i].replace(' ', '_').replace('/', '_')}"
                filepath = self.save_chart(subplot_filename, subplot_titles[i], new_fig)
                saved_files.append(filepath)
                
                # Close the new figure to free memory
                plt.close(new_fig)
        
        return saved_files
    
    def copy_subplot_content(self, source_ax, target_ax):
        """Copy content from source axes to target axes"""
        try:
            # Copy lines
            for line in source_ax.get_lines():
                target_ax.plot(line.get_xdata(), line.get_ydata(), 
                              color=line.get_color(), linewidth=line.get_linewidth(),
                              linestyle=line.get_linestyle(), marker=line.get_marker(),
                              markersize=line.get_markersize(), alpha=line.get_alpha(),
                              label=line.get_label())
            
            # Copy bar plots
            for patch in source_ax.patches:
                if hasattr(patch, 'get_height'):  # Bar patch
                    target_ax.bar(patch.get_x(), patch.get_height(), 
                                 width=patch.get_width(), color=patch.get_facecolor(),
                                 alpha=patch.get_alpha(), edgecolor=patch.get_edgecolor())
            
            # Copy collections (scatter plots, etc.)
            for collection in source_ax.collections:
                if hasattr(collection, 'get_offsets'):  # Scatter plot
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        target_ax.scatter(offsets[:, 0], offsets[:, 1], 
                                        c=collection.get_facecolors(),
                                        s=collection.get_sizes(),
                                        alpha=collection.get_alpha())
            
            # Copy images (heatmaps)
            for image in source_ax.get_images():
                target_ax.imshow(image.get_array(), cmap=image.get_cmap(),
                               extent=image.get_extent(), aspect='auto')
            
            # Copy labels and formatting
            target_ax.set_xlabel(source_ax.get_xlabel())
            target_ax.set_ylabel(source_ax.get_ylabel())
            target_ax.set_xlim(source_ax.get_xlim())
            target_ax.set_ylim(source_ax.get_ylim())
            
            # Copy tick labels
            target_ax.set_xticks(source_ax.get_xticks())
            target_ax.set_xticklabels(source_ax.get_xticklabels())
            target_ax.set_yticks(source_ax.get_yticks())
            target_ax.set_yticklabels(source_ax.get_yticklabels())
            
            # Copy legend if present
            if source_ax.get_legend():
                target_ax.legend()
            
            # Copy grid
            target_ax.grid(source_ax.get_gridlines() is not None)
            
        except Exception as e:
            self.log_text(f"⚠️ Warning: Could not copy all subplot content: {e}")
            # Fall back to copying basic properties
            target_ax.set_xlabel(source_ax.get_xlabel())
            target_ax.set_ylabel(source_ax.get_ylabel())
            target_ax.set_title(source_ax.get_title())
    
    def data_quality_assessment(self):
        """Enhanced data quality analysis with NaN handling and separate subplot saving"""
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
        
        # Save the combined dashboard
        self.save_chart("data_quality_dashboard", "Enhanced Data Quality with NaN Analysis", fig)
        
        # Save individual subplots separately
        subplot_titles = [
            'Missing Values Pattern',
            'Missing Values by Column',
            'Data Completeness by Column',
            'Overall Data Quality Score'
        ]
        
        self.log_text("\n📊 Saving individual subplots...")
        saved_files = self.save_subplot_separately(fig, axes, "data_quality", subplot_titles)
        
        for i, filepath in enumerate(saved_files):
            self.log_text(f"   📋 Subplot {i+1} saved: {os.path.basename(filepath)}")
        
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
        """Enhanced categorical analysis with NaN handling and separate subplot saving"""
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
            # Calculate subplot arrangement
            cols_per_row = 2
            rows = (n_cols + cols_per_row - 1) // cols_per_row
            
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 6 * rows))
            fig.suptitle('Categorical Data Distribution (Enhanced NaN Handling)', fontsize=16, fontweight='bold')
            
            # Handle different axes structures
            if n_cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            subplot_titles = []
            
            for i, col in enumerate(categorical_cols):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                
                if n_cols == 1:
                    ax = axes[0]
                elif rows == 1:
                    ax = axes[col_idx]
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
                
                title = f'{col} (Missing: {chart_data["cleaning_info"]["nan_count"]} values)'
                ax.set_title(title)
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                ax.set_xticks(range(len(chart_data['data'])))
                ax.set_xticklabels(chart_data['data'].index, rotation=45, ha='right')
                
                # Add value labels on bars
                for j, (bar, value) in enumerate(zip(bars, chart_data['data'].values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value}', ha='center', va='bottom', fontsize=9)
                
                subplot_titles.append(title)
            
            # Remove empty subplots
            total_subplots = rows * cols_per_row
            for i in range(n_cols, total_subplots):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                if rows == 1:
                    fig.delaxes(axes[col_idx])
                else:
                    fig.delaxes(axes[row, col_idx])
            
            plt.tight_layout()
            
            # Save the combined dashboard
            self.save_chart("categorical_dashboard", "Enhanced Categorical Analysis with NaN Handling", fig)
            
            # Save individual subplots separately
            self.log_text("\n📊 Saving individual categorical plots...")
            saved_files = self.save_subplot_separately(fig, axes, "categorical", subplot_titles)
            
            for i, filepath in enumerate(saved_files):
                self.log_text(f"   📋 Categorical plot {i+1} saved: {os.path.basename(filepath)}")
            
            plt.show()
        
        # Create individual detailed plots for each categorical column
        self.create_detailed_categorical_plots(categorical_cols)
    
    def create_detailed_categorical_plots(self, categorical_cols):
        """Create detailed individual plots for each categorical column"""
        self.log_text("\n📊 Creating detailed individual categorical plots...")
        
        for col in categorical_cols:
            # Prepare data
            chart_data = self.prepare_categorical_for_chart(self.df[col])
            
            # Create detailed figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Detailed Analysis: {col}', fontsize=16, fontweight='bold')
            
            # Plot 1: Bar chart
            bars = ax1.bar(range(len(chart_data['data'])), chart_data['data'].values)
            
            # Apply colors
            for j, (bar, color) in enumerate(zip(bars, chart_data['colors'])):
                if color is not None:
                    bar.set_color(color)
            
            ax1.set_title(f'Distribution of {col}')
            ax1.set_xlabel('Categories')
            ax1.set_ylabel('Count')
            ax1.set_xticks(range(len(chart_data['data'])))
            ax1.set_xticklabels(chart_data['data'].index, rotation=45, ha='right')
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, chart_data['data'].values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value}', ha='center', va='bottom', fontsize=10)
            
            # Plot 2: Pie chart
            colors_pie = ['red' if color is not None else 'lightblue' for color in chart_data['colors']]
            wedges, texts, autotexts = ax2.pie(chart_data['data'].values, 
                                              labels=chart_data['data'].index,
                                              autopct='%1.1f%%', 
                                              colors=colors_pie,
                                              startangle=90)
            
            ax2.set_title(f'Percentage Distribution of {col}')
            
            # Adjust text size for better readability
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            plt.tight_layout()
            
            # Save detailed plot
            detailed_filename = f"categorical_detailed_{col.replace(' ', '_').replace('/', '_')}"
            self.save_chart(detailed_filename, f"Detailed Analysis of {col}", fig)
            
            plt.show()
            plt.close(fig)
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
        """Enhanced numerical analysis with NaN handling and separate subplot saving"""
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
        
        # Create enhanced visualizations with separate subplot saving
        n_cols = len(numerical_cols)
        cols_per_row = 3
        rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 6 * rows))
        fig.suptitle('Numerical Data Distribution (NaN Values Excluded)', fontsize=16, fontweight='bold')
        
        # Handle different axes structures
        if n_cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        subplot_titles = []
        
        for i, col in enumerate(numerical_cols):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            
            if n_cols == 1:
                ax = axes[0]
            elif rows == 1:
                ax = axes[col_idx]
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
                title = f'{col} (Missing: {cleaning_info["nan_count"]}, {cleaning_info["nan_percentage"]:.1f}%)'
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
                title = f'{col} - No Data Available'
                ax.set_title(title)
            
            subplot_titles.append(title)
        
        # Remove empty subplots
        total_subplots = rows * cols_per_row
        for i in range(n_cols, total_subplots):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            if rows == 1:
                fig.delaxes(axes[col_idx])
            else:
                fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        
        # Save the combined dashboard
        self.save_chart("numerical_dashboard", "Enhanced Numerical Analysis with NaN Handling", fig)
        
        # Save individual subplots separately
        self.log_text("\n📊 Saving individual numerical plots...")
        saved_files = self.save_subplot_separately(fig, axes, "numerical", subplot_titles)
        
        for i, filepath in enumerate(saved_files):
            self.log_text(f"   📋 Numerical plot {i+1} saved: {os.path.basename(filepath)}")
        
        plt.show()
        
        # Create detailed individual plots for each numerical column
        self.create_detailed_numerical_plots(numerical_cols)
        
        # Correlation analysis with NaN handling
        if len(numerical_cols) > 1:
            self.create_correlation_analysis(numerical_cols)
    
    def create_detailed_numerical_plots(self, numerical_cols):
        """Create detailed individual plots for each numerical column"""
        self.log_text("\n📊 Creating detailed individual numerical plots...")
        
        for col in numerical_cols:
            # Clean data
            cleaned_series, cleaning_info = self.clean_data_for_visualization(self.df[col], 'numerical')
            
            if len(cleaned_series) == 0:
                continue
            
            # Create detailed figure with multiple views
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Detailed Analysis: {col}', fontsize=16, fontweight='bold')
            
            # Plot 1: Histogram
            ax1.hist(cleaned_series, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
            mean_val = cleaned_series.mean()
            median_val = cleaned_series.median()
            ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax1.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
            ax1.set_title(f'Distribution of {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            ax1.legend()
            
            # Plot 2: Box plot
            ax2.boxplot(cleaned_series, vert=True)
            ax2.set_title(f'Box Plot of {col}')
            ax2.set_ylabel(col)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Q-Q plot (normal distribution check)
            from scipy import stats
            stats.probplot(cleaned_series, dist="norm", plot=ax3)
            ax3.set_title(f'Q-Q Plot (Normal Distribution Check)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Statistics summary
            ax4.axis('off')
            stats_text = f"""
            Statistical Summary for {col}
            
            Count: {len(cleaned_series)}
            Missing: {cleaning_info['nan_count']} ({cleaning_info['nan_percentage']:.1f}%)
            
            Mean: {cleaned_series.mean():.4f}
            Median: {cleaned_series.median():.4f}
            Mode: {cleaned_series.mode().iloc[0] if len(cleaned_series.mode()) > 0 else 'N/A'}
            
            Std Dev: {cleaned_series.std():.4f}
            Variance: {cleaned_series.var():.4f}
            
            Min: {cleaned_series.min():.4f}
            25th Percentile: {cleaned_series.quantile(0.25):.4f}
            75th Percentile: {cleaned_series.quantile(0.75):.4f}
            Max: {cleaned_series.max():.4f}
            
            Skewness: {cleaned_series.skew():.4f}
            Kurtosis: {cleaned_series.kurtosis():.4f}
            """
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save detailed plot
            detailed_filename = f"numerical_detailed_{col.replace(' ', '_').replace('/', '_')}"
            self.save_chart(detailed_filename, f"Detailed Analysis of {col}", fig)
            
            plt.show()
            plt.close(fig)
    
    def create_correlation_analysis(self, numerical_cols):
        """Create correlation analysis with separate subplot saving"""
        self.log_text("\n🔗 Correlation Analysis (NaN values excluded):")
        
        # Create correlation matrix excluding NaN values
        clean_numeric_df = self.df[numerical_cols].dropna()
        
        if len(clean_numeric_df) == 0:
            self.log_text("   ⚠️ No complete cases available for correlation analysis")
            return
        
        correlation_matrix = clean_numeric_df.corr()
        
        # Create correlation visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Correlation Analysis (Based on {len(clean_numeric_df)} complete cases)', fontsize=16, fontweight='bold')
        
        # Plot 1: Full correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.3f', ax=ax1)
        ax1.set_title('Complete Correlation Matrix')
        
        # Plot 2: Masked correlation heatmap (upper triangle)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, mask=mask, fmt='.3f', ax=ax2)
        ax2.set_title('Correlation Matrix (Upper Triangle)')
        
        plt.tight_layout()
        
        # Save correlation analysis
        self.save_chart("correlation_analysis", "Enhanced Correlation Analysis", fig)
        
        # Save individual correlation plots
        subplot_titles = ['Complete Correlation Matrix', 'Correlation Matrix Upper Triangle']
        saved_files = self.save_subplot_separately(fig, [ax1, ax2], "correlation", subplot_titles)
        
        for i, filepath in enumerate(saved_files):
            self.log_text(f"   📋 Correlation plot {i+1} saved: {os.path.basename(filepath)}")
        
        plt.show()
        
        # Log missing data impact on correlation
        original_length = len(self.df)
        available_for_corr = len(clean_numeric_df)
        excluded_pct = ((original_length - available_for_corr) / original_length) * 100
        
        self.log_text(f"   Original rows: {original_length}")
        self.log_text(f"   Complete cases: {available_for_corr}")
        self.log_text(f"   Excluded due to NaN: {original_length - available_for_corr} ({excluded_pct:.1f}%)")
        
        # Find and report strong correlations
        self.log_text(f"\n📊 Strong Correlations (|r| > 0.7):")
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_val
                    ))
        
        if strong_correlations:
            for var1, var2, corr in strong_correlations:
                self.log_text(f"   {var1} ↔ {var2}: {corr:.3f}")
        else:
            self.log_text("   No strong correlations found")

    
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
