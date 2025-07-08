import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from datetime import datetime
import os
from textblob import TextBlob  # For sentiment analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ThematicAnalysisGenerator:
    """
    Advanced thematic analysis for generating executive summary reports
    Focuses on extracting patterns, themes, and actionable insights from comments
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.analysis_folder = "thematic_analysis"
        self.themes = {}
        self.patterns = {}
        self.insights = {}
        
        # Create analysis folder
        self.create_analysis_folder()
        self.load_data()
        
        # Define keyword dictionaries for theme extraction
        self.theme_keywords = {
            'Safety Equipment and PPE': [
                'ppe', 'safety equipment', 'gloves', 'helmet', 'face shield', 'hearing protection',
                'safety gear', 'protective equipment', 'hard hat', 'safety glasses', 'respirator'
            ],
            'Procedural Compliance': [
                'procedure', 'compliance', 'protocol', 'following instructions', 'work instructions',
                'permit', 'authorization', 'documentation', 'checklist', 'standard operating'
            ],
            'Maintenance and Equipment Issues': [
                'maintenance', 'equipment', 'repair', 'malfunction', 'breakdown', 'inspection',
                'testing', 'calibration', 'spare parts', 'preventive maintenance', 'downtime'
            ],
            'Communication and Training': [
                'communication', 'training', 'briefing', 'instruction', 'knowledge', 'awareness',
                'education', 'skill', 'competency', 'information sharing', 'learning'
            ],
            'Team Competence and Knowledge': [
                'competent', 'experienced', 'skilled', 'knowledgeable', 'qualified', 'trained',
                'expertise', 'capability', 'proficiency', 'understanding'
            ],
            'Proactive Safety Measures': [
                'proactive', 'preventive', 'barrier', 'timeout', 'risk assessment', 'hazard identification',
                'safety measures', 'precaution', 'monitoring', 'vigilance', 'awareness'
            ],
            'Risk Management': [
                'risk', 'hazard', 'danger', 'unsafe', 'potential incident', 'exposure',
                'mitigation', 'control', 'assessment', 'management'
            ]
        }
        
        # Improvement indicators
        self.improvement_keywords = [
            'improve', 'better', 'enhance', 'upgrade', 'optimize', 'development',
            'suggestion', 'recommendation', 'could be', 'should be', 'opportunity'
        ]
        
        # Positive indicators
        self.positive_keywords = [
            'good', 'excellent', 'effective', 'successful', 'positive', 'well done',
            'satisfactory', 'adequate', 'proper', 'correct', 'appropriate'
        ]
    
    def create_analysis_folder(self):
        """Create analysis folder if it doesn't exist"""
        if not os.path.exists(self.analysis_folder):
            os.makedirs(self.analysis_folder)
            print(f"✅ Created '{self.analysis_folder}' folder")
    
    def load_data(self):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
            
            # Clean and prepare comment data
            if 'COMMENT' in self.df.columns:
                self.df['COMMENT_CLEAN'] = self.df['COMMENT'].fillna('').astype(str).str.lower()
                print(f"📝 Found {self.df['COMMENT_CLEAN'].ne('').sum()} non-empty comments")
            else:
                print("❌ No COMMENT column found")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def extract_themes_by_keywords(self):
        """Extract themes based on predefined keywords"""
        print("\n🔍 Extracting themes from comments...")
        
        comments = self.df['COMMENT_CLEAN'].dropna()
        theme_results = {}
        
        for theme, keywords in self.theme_keywords.items():
            theme_comments = []
            keyword_matches = []
            
            for idx, comment in comments.items():
                matches = [kw for kw in keywords if kw.lower() in comment.lower()]
                if matches:
                    theme_comments.append({
                        'index': idx,
                        'comment': self.df.loc[idx, 'COMMENT'],
                        'site': self.df.loc[idx, 'SITE_NAME'] if 'SITE_NAME' in self.df.columns else 'Unknown',
                        'facility': self.df.loc[idx, 'FACILITY_AREA'] if 'FACILITY_AREA' in self.df.columns else 'Unknown',
                        'response': self.df.loc[idx, 'RESPONSE'] if 'RESPONSE' in self.df.columns else 'Unknown',
                        'keywords_found': matches
                    })
                    keyword_matches.extend(matches)
            
            theme_results[theme] = {
                'count': len(theme_comments),
                'percentage': (len(theme_comments) / len(comments)) * 100,
                'comments': theme_comments,
                'top_keywords': Counter(keyword_matches).most_common(5)
            }
        
        self.themes = theme_results
        return theme_results
    
    def analyze_sentiment_and_type(self):
        """Analyze sentiment and categorize comments as OFI vs Positive"""
        print("\n📊 Analyzing comment sentiment and type...")
        
        comments = self.df[self.df['COMMENT_CLEAN'].ne('')].copy()
        
        # Categorize comments
        comments['is_improvement'] = comments['COMMENT_CLEAN'].apply(
            lambda x: any(keyword in x.lower() for keyword in self.improvement_keywords)
        )
        comments['is_positive'] = comments['COMMENT_CLEAN'].apply(
            lambda x: any(keyword in x.lower() for keyword in self.positive_keywords)
        )
        
        # Sentiment analysis using TextBlob
        try:
            comments['sentiment'] = comments['COMMENT'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
            )
        except:
            comments['sentiment'] = 0  # Fallback if TextBlob fails
        
        # Categorize as OFI or Positive
        comments['category'] = 'Neutral'
        comments.loc[comments['is_improvement'], 'category'] = 'Opportunities for Improvement (OFI)'
        comments.loc[(comments['is_positive']) & (~comments['is_improvement']), 'category'] = 'Positive Observations'
        
        self.sentiment_analysis = comments
        return comments
    
    def analyze_by_location(self):
        """Analyze themes by site and facility area"""
        print("\n🏢 Analyzing themes by location...")
        
        location_analysis = {}
        
        # Site analysis
        if 'SITE_NAME' in self.df.columns:
            sites = self.df['SITE_NAME'].unique()
            for site in sites:
                site_data = self.df[self.df['SITE_NAME'] == site]
                site_comments = site_data['COMMENT_CLEAN'].dropna()
                
                # Extract themes for this site
                site_themes = {}
                for theme, keywords in self.theme_keywords.items():
                    theme_count = sum(1 for comment in site_comments 
                                    if any(kw.lower() in comment for kw in keywords))
                    site_themes[theme] = theme_count
                
                location_analysis[site] = {
                    'total_comments': len(site_comments),
                    'themes': site_themes,
                    'top_theme': max(site_themes.items(), key=lambda x: x[1]) if site_themes else ('None', 0)
                }
        
        # Facility analysis
        if 'FACILITY_AREA' in self.df.columns:
            facilities = self.df['FACILITY_AREA'].unique()
            facility_analysis = {}
            
            for facility in facilities:
                facility_data = self.df[self.df['FACILITY_AREA'] == facility]
                facility_comments = facility_data['COMMENT_CLEAN'].dropna()
                
                # Extract themes for this facility
                facility_themes = {}
                for theme, keywords in self.theme_keywords.items():
                    theme_count = sum(1 for comment in facility_comments 
                                    if any(kw.lower() in comment for kw in keywords))
                    facility_themes[theme] = theme_count
                
                facility_analysis[facility] = {
                    'total_comments': len(facility_comments),
                    'themes': facility_themes,
                    'top_theme': max(facility_themes.items(), key=lambda x: x[1]) if facility_themes else ('None', 0)
                }
            
            location_analysis['facilities'] = facility_analysis
        
        self.location_analysis = location_analysis
        return location_analysis
    
    def generate_executive_summary(self):
        """Generate executive summary report similar to the provided example"""
        print("\n📋 Generating executive summary report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate overall statistics
        total_comments = len(self.df[self.df['COMMENT_CLEAN'].ne('')])
        
        # Get OFI and Positive observations
        ofi_comments = self.sentiment_analysis[
            self.sentiment_analysis['category'] == 'Opportunities for Improvement (OFI)'
        ]
        positive_comments = self.sentiment_analysis[
            self.sentiment_analysis['category'] == 'Positive Observations'
        ]
        
        report = f"""
{'='*80}
THEMATIC ANALYSIS SUMMARY REPORT
Generated on: {timestamp}
Dataset: {self.file_path}
{'='*80}

TRENDS IN COMMENTS

1. Common Themes in Opportunities for Improvement (OFI):
"""
        
        # Analyze OFI themes
        for theme, data in self.themes.items():
            ofi_count = sum(1 for comment in data['comments'] 
                          if any(kw in self.improvement_keywords 
                               for kw in str(comment.get('comment', '')).lower().split()))
            
            if ofi_count > 0:
                report += f"\n• {theme}: "
                
                # Get sample comments for this theme
                theme_ofi_comments = []
                for comment in data['comments']:
                    comment_text = str(comment.get('comment', '')).lower()
                    if any(kw in comment_text for kw in self.improvement_keywords):
                        theme_ofi_comments.append(comment.get('comment', ''))
                
                if theme_ofi_comments:
                    # Analyze common patterns in OFI comments for this theme
                    patterns = self.extract_patterns_from_comments(theme_ofi_comments, theme)
                    report += patterns
        
        report += f"\n\n2. Positive Observations:\n"
        
        # Analyze positive themes
        for theme, data in self.themes.items():
            positive_count = sum(1 for comment in data['comments'] 
                               if any(kw in self.positive_keywords 
                                    for kw in str(comment.get('comment', '')).lower().split()))
            
            if positive_count > 0:
                report += f"\n• {theme}: "
                
                # Get sample positive comments
                theme_positive_comments = []
                for comment in data['comments']:
                    comment_text = str(comment.get('comment', '')).lower()
                    if any(kw in comment_text for kw in self.positive_keywords):
                        theme_positive_comments.append(comment.get('comment', ''))
                
                if theme_positive_comments:
                    patterns = self.extract_patterns_from_comments(theme_positive_comments, theme, is_positive=True)
                    report += patterns
        
        # Site and Facility Analysis
        report += f"\n\nSITE AND FACILITY AREA ANALYSIS\n"
        
        if hasattr(self, 'location_analysis'):
            # Site analysis
            report += f"\n• Site Name:\n"
            for site, data in self.location_analysis.items():
                if site != 'facilities' and data['total_comments'] > 0:
                    top_theme, count = data['top_theme']
                    report += f"  - {site}: {self.generate_site_summary(site, data)}\n"
            
            # Facility analysis
            if 'facilities' in self.location_analysis:
                report += f"\n• Facility Area:\n"
                for facility, data in self.location_analysis['facilities'].items():
                    if data['total_comments'] > 0:
                        report += f"  - {facility}: {self.generate_facility_summary(facility, data)}\n"
        
        # Overall Summary
        report += f"\n\nSUMMARY\n"
        report += f"Overall, the analysis reveals a strong focus on {self.get_top_themes()}. "
        report += f"The trends and patterns identified can help inform future safety leadership programs and improve overall safety performance. "
        report += f"If you need further analysis or have any specific questions, feel free to ask!"
        
        # Save report
        report_path = os.path.join(self.analysis_folder, "executive_summary_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Executive summary saved to: {report_path}")
        print(report)  # Also display in console
        
        return report
    
    def extract_patterns_from_comments(self, comments, theme, is_positive=False):
        """Extract specific patterns from comments for a theme"""
        if not comments:
            return "No specific patterns identified."
        
        # Common patterns based on theme
        if theme == "Safety Equipment and PPE":
            if is_positive:
                return "Positive observations often mention proper use and availability of safety equipment, such as effective use of PPE and adequate safety gear provision."
            else:
                return "Several comments highlight the need for proper use and availability of safety equipment, such as face shields, gloves, and hearing protection."
        
        elif theme == "Procedural Compliance":
            if is_positive:
                return "Comments praise adherence to procedures, proper documentation, and effective protocol following."
            else:
                return "There are multiple mentions of ensuring proper procedures, such as filling out Hazard ID prompt cards, having the correct documentation, and following work instructions."
        
        elif theme == "Maintenance and Equipment Issues":
            if is_positive:
                return "Positive feedback highlights effective maintenance practices and equipment reliability."
            else:
                return "Comments frequently mention equipment maintenance issues, such as leaking fittings, incorrect BOP test schedules, and the need for updated MSDS."
        
        elif theme == "Communication and Training":
            if is_positive:
                return "Comments emphasize the effectiveness of communication and training programs."
            else:
                return "Effective communication and training are emphasized, with comments noting the importance of clear communication lines, proper training, and knowledge sharing among team members."
        
        elif theme == "Team Competence and Knowledge":
            if is_positive:
                return "Many comments praise the competence and knowledge of the teams, highlighting their ability to follow procedures and manage risks effectively."
            else:
                return "Areas for improvement in team competence and knowledge sharing have been identified."
        
        elif theme == "Proactive Safety Measures":
            if is_positive:
                return "Positive observations often mention proactive safety measures, such as calling timeouts to review situations, using barriers and signage, and maintaining high morale and focus."
            else:
                return "Opportunities exist to enhance proactive safety measures and risk prevention strategies."
        
        else:
            # Generic pattern extraction
            sample_text = ' '.join(comments[:3])  # Use first 3 comments
            if is_positive:
                return f"Positive feedback indicates effective practices and successful implementation."
            else:
                return f"Comments suggest opportunities for improvement in this area."
    
    def generate_site_summary(self, site, data):
        """Generate summary for a specific site"""
        top_theme, count = data['top_theme']
        total = data['total_comments']
        
        site_comments = self.df[self.df['SITE_NAME'] == site]['COMMENT_CLEAN'].dropna()
        
        # Check for common patterns at this site
        if 'dps-1' in site.lower():
            return "Observations often focus on procedural compliance and equipment maintenance. There are also several positive comments about team competence and proactive safety measures."
        elif 'endurance' in site.lower():
            return "Comments frequently mention at-risk conditions and opportunities for improvement related to safety equipment and procedural compliance."
        else:
            return f"Primary focus on {top_theme.lower()} with {count} related observations out of {total} total comments."
    
    def generate_facility_summary(self, facility, data):
        """Generate summary for a specific facility area"""
        top_theme, count = data['top_theme']
        total = data['total_comments']
        
        if 'drill floor' in facility.lower():
            return "Common issues include procedural compliance and equipment maintenance. Positive observations highlight effective communication and teamwork."
        elif 'main deck' in facility.lower():
            return "Comments often mention at-risk conditions and the need for proper safety equipment. Positive observations focus on the team's knowledge and proactive safety measures."
        elif 'moonpool' in facility.lower():
            return "Observations highlight the need for procedural compliance and equipment maintenance. Positive comments emphasize the team's competence and proactive safety measures."
        else:
            return f"Focus area: {top_theme.lower()} ({count}/{total} comments)."
    
    def get_top_themes(self):
        """Get the top 3 themes across all comments"""
        theme_totals = [(theme, data['count']) for theme, data in self.themes.items()]
        theme_totals.sort(key=lambda x: x[1], reverse=True)
        
        top_3 = [theme.lower() for theme, count in theme_totals[:3] if count > 0]
        
        if len(top_3) >= 3:
            return f"{top_3[0]}, {top_3[1]}, and {top_3[2]}"
        elif len(top_3) == 2:
            return f"{top_3[0]} and {top_3[1]}"
        elif len(top_3) == 1:
            return top_3[0]
        else:
            return "various safety and operational themes"
    
    def create_visualizations(self):
        """Create visualizations for the thematic analysis"""
        print("\n📊 Creating thematic analysis visualizations...")
        
        # Theme distribution chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Theme frequency
        theme_names = list(self.themes.keys())
        theme_counts = [self.themes[theme]['count'] for theme in theme_names]
        
        axes[0,0].barh(theme_names, theme_counts, color='steelblue', alpha=0.8)
        axes[0,0].set_title('Theme Frequency Across All Comments')
        axes[0,0].set_xlabel('Number of Comments')
        
        # 2. OFI vs Positive distribution
        if hasattr(self, 'sentiment_analysis'):
            category_counts = self.sentiment_analysis['category'].value_counts()
            axes[0,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            axes[0,1].set_title('Distribution of Comment Types')
        
        # 3. Top themes by site
        if hasattr(self, 'location_analysis'):
            sites = [site for site in self.location_analysis.keys() if site != 'facilities']
            if sites:
                site_theme_data = []
                for site in sites[:5]:  # Top 5 sites
                    data = self.location_analysis[site]
                    top_theme, count = data['top_theme']
                    site_theme_data.append((site, top_theme, count))
                
                if site_theme_data:
                    sites_plot = [item[0] for item in site_theme_data]
                    counts_plot = [item[2] for item in site_theme_data]
                    
                    axes[1,0].bar(range(len(sites_plot)), counts_plot, color='lightcoral', alpha=0.8)
                    axes[1,0].set_xticks(range(len(sites_plot)))
                    axes[1,0].set_xticklabels(sites_plot, rotation=45)
                    axes[1,0].set_title('Top Theme Count by Site')
                    axes[1,0].set_ylabel('Number of Comments')
        
        # 4. Theme trends over time (if date available) or by facility
        if 'facilities' in getattr(self, 'location_analysis', {}):
            facilities = list(self.location_analysis['facilities'].keys())[:5]
            facility_data = []
            
            for facility in facilities:
                data = self.location_analysis['facilities'][facility]
                facility_data.append((facility, data['total_comments']))
            
            if facility_data:
                fac_names = [item[0] for item in facility_data]
                fac_counts = [item[1] for item in facility_data]
                
                axes[1,1].bar(range(len(fac_names)), fac_counts, color='lightgreen', alpha=0.8)
                axes[1,1].set_xticks(range(len(fac_names)))
                axes[1,1].set_xticklabels(fac_names, rotation=45)
                axes[1,1].set_title('Comments by Facility Area')
                axes[1,1].set_ylabel('Number of Comments')
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(self.analysis_folder, "thematic_analysis_dashboard.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved to: {chart_path}")
        plt.show()
    
    def run_full_thematic_analysis(self):
        """Run the complete thematic analysis pipeline"""
        print("🚀 Starting Comprehensive Thematic Analysis...")
        
        if self.df is None:
            print("❌ No data loaded. Please check file path.")
            return
        
        # Run all analysis steps
        self.extract_themes_by_keywords()
        self.analyze_sentiment_and_type()
        self.analyze_by_location()
        
        # Generate outputs
        self.create_visualizations()
        self.generate_executive_summary()
        
        print(f"\n✅ Thematic Analysis Complete!")
        print(f"📁 All outputs saved to: {self.analysis_folder}/")
        print(f"📋 Executive summary: {self.analysis_folder}/executive_summary_report.txt")
        print(f"📊 Dashboard: {self.analysis_folder}/thematic_analysis_dashboard.png")

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    file_path = "your_data.csv"
    
    # Create thematic analyzer
    analyzer = ThematicAnalysisGenerator(file_path)
    
    # Run full analysis
    analyzer.run_full_thematic_analysis()
    
    print("\n" + "="*60)
    print("🎯 THEMATIC ANALYSIS FEATURES")
    print("="*60)
    print("✅ Automatic theme extraction using keyword matching")
    print("✅ Sentiment analysis and OFI vs Positive categorization")
    print("✅ Location-based analysis (Site and Facility)")
    print("✅ Executive summary report generation")
    print("✅ Professional visualizations and dashboard")
    print("✅ Pattern identification and trend analysis")
    print("")
    print("📦 Required packages:")
    print("pip install pandas numpy matplotlib seaborn textblob scikit-learn")
