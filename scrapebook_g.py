import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import base64
import io
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

class SafetyAnalysisHTMLReportGenerator:
    """
    Generate professional HTML reports for safety inspection comment analysis
    """
    
    def __init__(self, df, claude_analysis=None, report_title="Safety Inspection Analysis Report"):
        """
        Initialize HTML report generator
        
        Args:
            df: DataFrame with safety inspection data
            claude_analysis: Results from Claude analysis (optional)
            report_title: Title for the report
        """
        self.df = df
        self.claude_analysis = claude_analysis
        self.report_title = report_title
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Report styling configuration
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Initialize chart counter
        self.chart_counter = 1
        
    def matplotlib_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def plotly_to_html(self, fig):
        """Convert plotly figure to HTML div"""
        return pio.to_html(fig, include_plotlyjs='cdn', div_id=f'plotly-div-{self.chart_counter}')
    
    def create_executive_summary_section(self):
        """Create executive summary section"""
        
        # Calculate key metrics
        total_records = len(self.df)
        date_range = f"{self.df['INSPECTION_DATE'].min().strftime('%Y-%m-%d')} to {self.df['INSPECTION_DATE'].max().strftime('%Y-%m-%d')}"
        unique_sites = self.df['SITE_NAME'].nunique() if 'SITE_NAME' in self.df.columns else 0
        unique_facilities = self.df['FACILITY_AREA'].nunique() if 'FACILITY_AREA' in self.df.columns else 0
        
        # Response distribution
        response_dist = self.df['RESPONSE'].value_counts() if 'RESPONSE' in self.df.columns else pd.Series()
        
        # Calculate effectiveness rate
        if 'RESPONSE' in self.df.columns:
            effective_count = self.df[self.df['RESPONSE'] == 'Effective'].shape[0]
            effectiveness_rate = (effective_count / total_records) * 100
        else:
            effectiveness_rate = 0
        
        # Comment availability
        comment_availability = (self.df['COMMENT'].notna().sum() / total_records) * 100
        
        summary_html = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-chart-line"></i> Executive Summary
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5 class="text-primary">Dataset Overview</h5>
                                <ul class="list-unstyled">
                                    <li><strong>Total Records:</strong> {total_records:,}</li>
                                    <li><strong>Date Range:</strong> {date_range}</li>
                                    <li><strong>Unique Sites:</strong> {unique_sites}</li>
                                    <li><strong>Unique Facility Areas:</strong> {unique_facilities}</li>
                                    <li><strong>Comment Availability:</strong> {comment_availability:.1f}%</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5 class="text-primary">Key Performance Indicators</h5>
                                <div class="row">
                                    <div class="col-6">
                                        <div class="text-center p-3 bg-light rounded">
                                            <h4 class="text-success">{effectiveness_rate:.1f}%</h4>
                                            <small>Effectiveness Rate</small>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="text-center p-3 bg-light rounded">
                                            <h4 class="text-info">{len(response_dist)}</h4>
                                            <small>Response Types</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return summary_html
    
    def create_data_quality_section(self):
        """Create data quality assessment section"""
        
        # Missing data analysis
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        # Create missing data visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Missing data bar chart
        missing_data_plot = missing_data[missing_data > 0]
        if len(missing_data_plot) > 0:
            bars = ax1.bar(range(len(missing_data_plot)), missing_data_plot.values, 
                          color=self.colors['warning'], alpha=0.8)
            ax1.set_title('Missing Data by Column', fontweight='bold')
            ax1.set_xlabel('Columns')
            ax1.set_ylabel('Missing Count')
            ax1.set_xticks(range(len(missing_data_plot)))
            ax1.set_xticklabels(missing_data_plot.index, rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, missing_data_plot.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data!', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=16, color='green', weight='bold')
            ax1.set_title('Data Completeness Status', fontweight='bold')
        
        # Data completeness pie chart
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = missing_data.sum()
        complete_cells = total_cells - missing_cells
        
        sizes = [complete_cells, missing_cells]
        labels = ['Complete Data', 'Missing Data']
        colors = [self.colors['success'], self.colors['danger']]
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Data Quality', fontweight='bold')
        
        plt.tight_layout()
        data_quality_img = self.matplotlib_to_base64(fig)
        
        # Create data quality table
        quality_table_html = """
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead class="table-dark">
                    <tr>
                        <th>Column</th>
                        <th>Total Records</th>
                        <th>Missing</th>
                        <th>Missing %</th>
                        <th>Data Type</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for col in self.df.columns:
            missing_count = missing_data[col]
            missing_percentage = missing_pct[col]
            data_type = str(self.df[col].dtype)
            
            if missing_percentage == 0:
                status = '<span class="badge bg-success">Complete</span>'
            elif missing_percentage < 10:
                status = '<span class="badge bg-warning">Good</span>'
            elif missing_percentage < 30:
                status = '<span class="badge bg-warning">Moderate</span>'
            else:
                status = '<span class="badge bg-danger">Poor</span>'
            
            quality_table_html += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td>{len(self.df):,}</td>
                    <td>{missing_count}</td>
                    <td>{missing_percentage:.1f}%</td>
                    <td><code>{data_type}</code></td>
                    <td>{status}</td>
                </tr>
            """
        
        quality_table_html += """
                </tbody>
            </table>
        </div>
        """
        
        data_quality_html = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-info">
                    <div class="card-header bg-info text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-check-circle"></i> Data Quality Assessment
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="row mb-4">
                            <div class="col-12">
                                <img src="{data_quality_img}" class="img-fluid" alt="Data Quality Charts">
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                <h5 class="text-info mb-3">Detailed Data Quality Metrics</h5>
                                {quality_table_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return data_quality_html
    
    def create_response_analysis_section(self):
        """Create response pattern analysis section"""
        
        if 'RESPONSE' not in self.df.columns:
            return '<div class="alert alert-warning">Response column not available for analysis.</div>'
        
        # Response distribution
        response_counts = self.df['RESPONSE'].value_counts()
        response_pct = self.df['RESPONSE'].value_counts(normalize=True) * 100
        
        # Create interactive plotly charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Distribution', 'Response by Site', 
                           'Response Timeline', 'Response by Facility Area'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Response distribution pie chart
        fig.add_trace(
            go.Pie(labels=response_counts.index, values=response_counts.values,
                   name="Response Distribution"),
            row=1, col=1
        )
        
        # 2. Response by site
        if 'SITE_NAME' in self.df.columns:
            site_response = pd.crosstab(self.df['SITE_NAME'], self.df['RESPONSE'])
            for response_type in site_response.columns:
                fig.add_trace(
                    go.Bar(name=response_type, x=site_response.index, 
                           y=site_response[response_type]),
                    row=1, col=2
                )
        
        # 3. Response timeline
        if 'INSPECTION_DATE' in self.df.columns:
            timeline_data = self.df.groupby([self.df['INSPECTION_DATE'].dt.date, 'RESPONSE']).size().reset_index(name='count')
            for response_type in timeline_data['RESPONSE'].unique():
                subset = timeline_data[timeline_data['RESPONSE'] == response_type]
                fig.add_trace(
                    go.Scatter(x=subset['INSPECTION_DATE'], y=subset['count'],
                              mode='lines+markers', name=response_type),
                    row=2, col=1
                )
        
        # 4. Response by facility area
        if 'FACILITY_AREA' in self.df.columns:
            facility_response = pd.crosstab(self.df['FACILITY_AREA'], self.df['RESPONSE'])
            facility_top = facility_response.sum(axis=1).nlargest(10)
            facility_response_top = facility_response.loc[facility_top.index]
            
            for response_type in facility_response_top.columns:
                fig.add_trace(
                    go.Bar(name=response_type, x=facility_response_top.index, 
                           y=facility_response_top[response_type]),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, title_text="Response Pattern Analysis")
        response_charts_html = self.plotly_to_html(fig)
        self.chart_counter += 1
        
        # Response summary table
        response_table_html = """
        <div class="table-responsive">
            <table class="table table-striped">
                <thead class="table-dark">
                    <tr>
                        <th>Response Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for response, count in response_counts.items():
            percentage = response_pct[response]
            
            if response == 'Effective':
                status = '<span class="badge bg-success">Positive</span>'
            elif response == 'Ineffective':
                status = '<span class="badge bg-danger">Needs Attention</span>'
            else:
                status = '<span class="badge bg-secondary">Other</span>'
            
            response_table_html += f"""
                <tr>
                    <td><strong>{response}</strong></td>
                    <td>{count:,}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{status}</td>
                </tr>
            """
        
        response_table_html += """
                </tbody>
            </table>
        </div>
        """
        
        response_html = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-success">
                    <div class="card-header bg-success text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-chart-bar"></i> Response Pattern Analysis
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="row mb-4">
                            <div class="col-md-6">
                                {response_table_html}
                            </div>
                            <div class="col-md-6">
                                <div class="alert alert-info">
                                    <h6 class="alert-heading">Key Insights</h6>
                                    <ul class="mb-0">
                                        <li>Total responses analyzed: {len(self.df):,}</li>
                                        <li>Most common response: {response_counts.index[0]} ({response_pct.iloc[0]:.1f}%)</li>
                                        <li>Response types identified: {len(response_counts)}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                {response_charts_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return response_html
    
    def create_claude_analysis_section(self):
        """Create Claude analysis results section"""
        
        if not self.claude_analysis:
            return f"""
            <div class="row mb-4">
                <div class="col-12">
                    <div class="alert alert-warning">
                        <h5 class="alert-heading">
                            <i class="fas fa-robot"></i> AI Analysis Not Available
                        </h5>
                        <p>Claude AI analysis was not provided or failed to complete. The report includes statistical analysis only.</p>
                        <hr>
                        <p class="mb-0">To get AI-powered insights, run the Claude analysis component and regenerate this report.</p>
                    </div>
                </div>
            </div>
            """
        
        # Extract Claude analysis content
        if isinstance(self.claude_analysis, dict):
            analysis_content = self.claude_analysis.get('analysis', 'No analysis content available')
        else:
            analysis_content = str(self.claude_analysis)
        
        # Process the analysis content to create structured HTML
        analysis_html = self.format_claude_analysis(analysis_content)
        
        claude_section = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-warning">
                    <div class="card-header bg-warning text-dark">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-robot"></i> AI-Powered Comment Analysis
                        </h3>
                    </div>
                    <div class="card-body">
                        {analysis_html}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return claude_section
    
    def format_claude_analysis(self, analysis_content):
        """Format Claude analysis content into structured HTML"""
        
        # Split content into sections
        lines = analysis_content.split('\n')
        formatted_html = ""
        current_section = ""
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Check for headers (lines with ** or numbered sections)
            if line.startswith('**') and line.endswith('**'):
                if in_list:
                    formatted_html += "</ul>\n"
                    in_list = False
                header_text = line.replace('**', '')
                formatted_html += f'<h5 class="text-warning mt-4 mb-3"><i class="fas fa-chevron-right"></i> {header_text}</h5>\n'
                
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if in_list:
                    formatted_html += "</ul>\n"
                    in_list = False
                formatted_html += f'<h6 class="text-primary mt-3 mb-2">{line}</h6>\n'
                
            elif line.startswith('-') or line.startswith('•'):
                if not in_list:
                    formatted_html += '<ul class="list-group list-group-flush mb-3">\n'
                    in_list = True
                list_item = line.replace('-', '').replace('•', '').strip()
                
                # Check if line contains quotes (exact examples)
                if '"' in list_item:
                    formatted_html += f'<li class="list-group-item"><i class="fas fa-quote-left text-muted"></i> {list_item}</li>\n'
                else:
                    formatted_html += f'<li class="list-group-item">{list_item}</li>\n'
                    
            else:
                if in_list:
                    formatted_html += "</ul>\n"
                    in_list = False
                
                # Check if line contains quotes (exact examples)
                if '"' in line:
                    formatted_html += f'<div class="alert alert-light border-left-primary"><i class="fas fa-quote-left text-muted"></i> {line}</div>\n'
                else:
                    formatted_html += f'<p class="mb-2">{line}</p>\n'
        
        if in_list:
            formatted_html += "</ul>\n"
        
        return formatted_html
    
    def create_site_facility_section(self):
        """Create site and facility analysis section"""
        
        # Site analysis
        site_analysis_html = ""
        if 'SITE_NAME' in self.df.columns:
            site_stats = self.df['SITE_NAME'].value_counts()
            
            # Create site distribution chart
            fig_site = px.bar(
                x=site_stats.index, 
                y=site_stats.values,
                title="Inspection Activity by Site",
                labels={'x': 'Site Name', 'y': 'Number of Inspections'},
                color=site_stats.values,
                color_continuous_scale='Blues'
            )
            fig_site.update_layout(height=400, showlegend=False)
            site_chart_html = self.plotly_to_html(fig_site)
            self.chart_counter += 1
            
            site_analysis_html = f"""
            <div class="row mb-4">
                <div class="col-md-6">
                    <h5 class="text-primary">Site Activity Summary</h5>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead class="table-dark">
                                <tr><th>Site</th><th>Inspections</th><th>Percentage</th></tr>
                            </thead>
                            <tbody>
            """
            
            for site, count in site_stats.head(10).items():
                percentage = (count / len(self.df)) * 100
                site_analysis_html += f"""
                    <tr>
                        <td><strong>{site}</strong></td>
                        <td>{count:,}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                """
            
            site_analysis_html += f"""
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-6">
                    {site_chart_html}
                </div>
            </div>
            """
        
        # Facility analysis
        facility_analysis_html = ""
        if 'FACILITY_AREA' in self.df.columns:
            facility_stats = self.df['FACILITY_AREA'].value_counts()
            
            # Create facility distribution chart
            fig_facility = px.pie(
                values=facility_stats.values[:10], 
                names=facility_stats.index[:10],
                title="Top 10 Facility Areas by Inspection Volume"
            )
            fig_facility.update_layout(height=400)
            facility_chart_html = self.plotly_to_html(fig_facility)
            self.chart_counter += 1
            
            facility_analysis_html = f"""
            <div class="row mb-4">
                <div class="col-md-6">
                    {facility_chart_html}
                </div>
                <div class="col-md-6">
                    <h5 class="text-primary">Facility Area Summary</h5>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead class="table-dark">
                                <tr><th>Facility Area</th><th>Inspections</th><th>Percentage</th></tr>
                            </thead>
                            <tbody>
            """
            
            for facility, count in facility_stats.head(10).items():
                percentage = (count / len(self.df)) * 100
                facility_analysis_html += f"""
                    <tr>
                        <td><strong>{facility}</strong></td>
                        <td>{count:,}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                """
            
            facility_analysis_html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            """
        
        site_facility_html = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-secondary">
                    <div class="card-header bg-secondary text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-map-marker-alt"></i> Site & Facility Analysis
                        </h3>
                    </div>
                    <div class="card-body">
                        {site_analysis_html}
                        {facility_analysis_html}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return site_facility_html
    
    def create_timeline_section(self):
        """Create inspection timeline analysis section"""
        
        if 'INSPECTION_DATE' not in self.df.columns:
            return '<div class="alert alert-warning">Date column not available for timeline analysis.</div>'
        
        # Prepare date data
        date_data = self.df['INSPECTION_DATE'].dropna()
        
        if len(date_data) == 0:
            return '<div class="alert alert-warning">No valid dates found for timeline analysis.</div>'
        
        # Monthly timeline
        monthly_counts = date_data.dt.to_period('M').value_counts().sort_index()
        
        # Create timeline chart
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=monthly_counts.index.to_timestamp(),
            y=monthly_counts.values,
            mode='lines+markers',
            name='Monthly Inspections',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=8)
        ))
        
        fig_timeline.update_layout(
            title='Inspection Activity Timeline',
            xaxis_title='Month',
            yaxis_title='Number of Inspections',
            height=400,
            hovermode='x unified'
        )
        
        timeline_chart_html = self.plotly_to_html(fig_timeline)
        self.chart_counter += 1
        
        # Day of week analysis
        daily_counts = date_data.dt.day_name().value_counts()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = daily_counts.reindex([day for day in day_order if day in daily_counts.index])
        
        fig_daily = px.bar(
            x=daily_counts.index,
            y=daily_counts.values,
            title='Inspection Activity by Day of Week',
            labels={'x': 'Day of Week', 'y': 'Number of Inspections'},
            color=daily_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_daily.update_layout(height=400, showlegend=False)
        daily_chart_html = self.plotly_to_html(fig_daily)
        self.chart_counter += 1
        
        timeline_html = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-info">
                    <div class="card-header bg-info text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-calendar-alt"></i> Inspection Timeline Analysis
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="row mb-4">
                            <div class="col-12">
                                {timeline_chart_html}
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                {daily_chart_html}
                            </div>
                        </div>
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <div class="alert alert-info">
                                    <h6 class="alert-heading">Timeline Insights</h6>
                                    <ul class="mb-0">
                                        <li>Date range: {date_data.min().strftime('%Y-%m-%d')} to {date_data.max().strftime('%Y-%m-%d')}</li>
                                        <li>Peak month: {monthly_counts.idxmax().strftime('%Y-%m')} ({monthly_counts.max()} inspections)</li>
                                        <li>Most active day: {daily_counts.idxmax()} ({daily_counts.max()} inspections)</li>
                                    </ul>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="alert alert-secondary">
                                    <h6 class="alert-heading">Activity Statistics</h6>
                                    <ul class="mb-0">
                                        <li>Average per month: {monthly_counts.mean():.1f} inspections</li>
                                        <li>Total months covered: {len(monthly_counts)}</li>
                                        <li>Active days per week: {len(daily_counts)}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return timeline_html
    
    def create_recommendations_section(self):
        """Create recommendations and next steps section"""
        
        # Calculate metrics for recommendations
        total_records = len(self.df)
        effectiveness_rate = 0
        comment_rate = 0
        
        if 'RESPONSE' in self.df.columns:
            effective_count = self.df[self.df['RESPONSE'] == 'Effective'].shape[0]
            effectiveness_rate = (effective_count / total_records) * 100
        
        if 'COMMENT' in self.df.columns:
            comment_count = self.df['COMMENT'].notna().sum()
            comment_rate = (comment_count / total_records) * 100
        
        # Generate recommendations based on data
        recommendations = []
        
        if effectiveness_rate < 70:
            recommendations.append({
                'type': 'warning',
                'title': 'Effectiveness Rate Below Target',
                'description': f'Current effectiveness rate is {effectiveness_rate:.1f}%. Consider reviewing inspection procedures and training.',
                'action': 'Implement targeted training programs and review inspection protocols.'
            })
        else:
            recommendations.append({
                'type': 'success',
                'title': 'Good Effectiveness Rate',
                'description': f'Effectiveness rate of {effectiveness_rate:.1f}% indicates strong safety practices.',
                'action': 'Maintain current standards and share best practices across sites.'
            })
        
        if comment_rate < 80:
            recommendations.append({
                'type': 'info',
                'title': 'Improve Comment Documentation',
                'description': f'Only {comment_rate:.1f}% of inspections have detailed comments.',
                'action': 'Encourage inspectors to provide more detailed observations and findings.'
            })
        
        # Site-specific recommendations
        if 'SITE_NAME' in self.df.columns:
            site_response_analysis = pd.crosstab(self.df['SITE_NAME'], self.df['RESPONSE'], normalize='index') * 100
            if 'Ineffective' in site_response_analysis.columns:
                problematic_sites = site_response_analysis[site_response_analysis['Ineffective'] > 30]
                if len(problematic_sites) > 0:
                    recommendations.append({
                        'type': 'danger',
                        'title': 'Sites Requiring Attention',
                        'description': f'{len(problematic_sites)} sites have >30% ineffective responses.',
                        'action': f'Focus improvement efforts on: {", ".join(problematic_sites.index[:3])}'
                    })
        
        recommendations_html = """
        <div class="row">
        """
        
        for rec in recommendations:
            alert_class = f"alert-{rec['type']}"
            icon_map = {
                'success': 'check-circle',
                'warning': 'exclamation-triangle', 
                'danger': 'exclamation-circle',
                'info': 'info-circle'
            }
            icon = icon_map.get(rec['type'], 'info-circle')
            
            recommendations_html += f"""
            <div class="col-md-6 mb-3">
                <div class="alert {alert_class}">
                    <h6 class="alert-heading">
                        <i class="fas fa-{icon}"></i> {rec['title']}
                    </h6>
                    <p class="mb-2">{rec['description']}</p>
                    <hr>
                    <p class="mb-0"><strong>Recommended Action:</strong> {rec['action']}</p>
                </div>
            </div>
            """
        
        recommendations_html += """
        </div>
        """
        
        next_steps_html = """
        <div class="row mt-4">
            <div class="col-12">
                <div class="card border-dark">
                    <div class="card-header bg-dark text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-tasks"></i> Next Steps & Action Items
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <h6 class="text-primary">Immediate Actions (1-30 days)</h6>
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item">Review ineffective inspection findings</li>
                                    <li class="list-group-item">Conduct follow-up inspections at high-risk sites</li>
                                    <li class="list-group-item">Update inspection documentation standards</li>
                                </ul>
                            </div>
                            <div class="col-md-4">
                                <h6 class="text-warning">Short-term Actions (1-3 months)</h6>
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item">Implement additional training programs</li>
                                    <li class="list-group-item">Establish regular monitoring cycles</li>
                                    <li class="list-group-item">Develop site-specific improvement plans</li>
                                </ul>
                            </div>
                            <div class="col-md-4">
                                <h6 class="text-success">Long-term Goals (3-12 months)</h6>
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item">Achieve >85% effectiveness rate</li>
                                    <li class="list-group-item">Standardize best practices across all sites</li>
                                    <li class="list-group-item">Implement predictive safety analytics</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        recommendations_section = f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-lightbulb"></i> Recommendations & Action Plan
                        </h3>
                    </div>
                    <div class="card-body">
                        {recommendations_html}
                        {next_steps_html}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return recommendations_section
    
    def generate_full_html_report(self, output_file=None):
        """Generate the complete HTML report"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"safety_inspection_report_{timestamp}.html"
        
        # Generate all sections
        print("📊 Generating report sections...")
        
        executive_summary = self.create_executive_summary_section()
        data_quality = self.create_data_quality_section()
        response_analysis = self.create_response_analysis_section()
        claude_analysis = self.create_claude_analysis_section()
        site_facility = self.create_site_facility_section()
        timeline = self.create_timeline_section()
        recommendations = self.create_recommendations_section()
        
        # Create the complete HTML document
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.report_title}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Custom Styles -->
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        
        .navbar-brand {{
            font-weight: bold;
            font-size: 1.5rem;
        }}
        
        .card {{
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border: 1px solid rgba(0, 0, 0, 0.125);
            margin-bottom: 1.5rem;
        }}
        
        .card-header {{
            border-bottom: 1px solid rgba(0, 0, 0, 0.125);
            font-weight: 600;
        }}
        
        .alert {{
            border: 1px solid transparent;
            border-radius: 0.375rem;
        }}
        
        .table {{
            margin-bottom: 0;
        }}
        
        .badge {{
            font-size: 0.75em;
        }}
        
        .text-primary {{ color: {self.colors['primary']} !important; }}
        .text-secondary {{ color: {self.colors['secondary']} !important; }}
        .bg-primary {{ background-color: {self.colors['primary']} !important; }}
        .bg-secondary {{ background-color: {self.colors['secondary']} !important; }}
        .border-primary {{ border-color: {self.colors['primary']} !important; }}
        .border-secondary {{ border-color: {self.colors['secondary']} !important; }}
        
        .report-header {{
            background: linear-gradient(135deg, {self.colors['primary']}, {self.colors['secondary']});
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }}
        
        .section-divider {{
            border-top: 2px solid {self.colors['primary']};
            margin: 2rem 0;
        }}
        
        @media print {{
            .no-print {{ display: none !important; }}
            .card {{ page-break-inside: avoid; }}
        }}
        
        .toc {{
            background-color: white;
            border-radius: 0.375rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .toc a {{
            text-decoration: none;
            color: {self.colors['primary']};
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark" style="background-color: {self.colors['primary']};">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-shield-alt"></i> Safety Inspection Report
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text">
                    Generated: {self.timestamp}
                </span>
            </div>
        </div>
    </nav>

    <!-- Report Header -->
    <div class="report-header">
        <div class="container">
            <div class="row">
                <div class="col-12 text-center">
                    <h1 class="display-4 mb-3">
                        <i class="fas fa-clipboard-check"></i> {self.report_title}
                    </h1>
                    <p class="lead mb-0">
                        Comprehensive Analysis of Safety Inspection Data
                    </p>
                    <p class="mb-0">
                        Report Generated: {self.timestamp}
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="container">
        
        <!-- Table of Contents -->
        <div class="row">
            <div class="col-12">
                <div class="toc">
                    <h4 class="text-primary mb-3">
                        <i class="fas fa-list"></i> Table of Contents
                    </h4>
                    <div class="row">
                        <div class="col-md-6">
                            <ul class="list-unstyled">
                                <li><a href="#executive-summary"><i class="fas fa-chart-line"></i> Executive Summary</a></li>
                                <li><a href="#data-quality"><i class="fas fa-check-circle"></i> Data Quality Assessment</a></li>
                                <li><a href="#response-analysis"><i class="fas fa-chart-bar"></i> Response Pattern Analysis</a></li>
                                <li><a href="#ai-analysis"><i class="fas fa-robot"></i> AI-Powered Analysis</a></li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <ul class="list-unstyled">
                                <li><a href="#site-facility"><i class="fas fa-map-marker-alt"></i> Site & Facility Analysis</a></li>
                                <li><a href="#timeline"><i class="fas fa-calendar-alt"></i> Timeline Analysis</a></li>
                                <li><a href="#recommendations"><i class="fas fa-lightbulb"></i> Recommendations</a></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div id="executive-summary">
            {executive_summary}
        </div>
        
        <div class="section-divider"></div>

        <!-- Data Quality Assessment -->
        <div id="data-quality">
            {data_quality}
        </div>
        
        <div class="section-divider"></div>

        <!-- Response Analysis -->
        <div id="response-analysis">
            {response_analysis}
        </div>
        
        <div class="section-divider"></div>

        <!-- Claude Analysis -->
        <div id="ai-analysis">
            {claude_analysis}
        </div>
        
        <div class="section-divider"></div>

        <!-- Site & Facility Analysis -->
        <div id="site-facility">
            {site_facility}
        </div>
        
        <div class="section-divider"></div>

        <!-- Timeline Analysis -->
        <div id="timeline">
            {timeline}
        </div>
        
        <div class="section-divider"></div>

        <!-- Recommendations -->
        <div id="recommendations">
            {recommendations}
        </div>
        
        <!-- Footer -->
        <div class="row mt-5">
            <div class="col-12">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <p class="mb-2">
                            <strong>Report Information</strong>
                        </p>
                        <p class="mb-1">
                            Generated on {self.timestamp} | 
                            Total Records: {len(self.df):,} | 
                            Data Range: {self.df['INSPECTION_DATE'].min().strftime('%Y-%m-%d') if 'INSPECTION_DATE' in self.df.columns else 'N/A'} to {self.df['INSPECTION_DATE'].max().strftime('%Y-%m-%d') if 'INSPECTION_DATE' in self.df.columns else 'N/A'}
                        </p>
                        <p class="mb-0 text-muted">
                            This report was generated using automated data analysis and AI-powered insights.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Print functionality -->
    <script>
        function printReport() {{
            window.print();
        }}
        
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
    </script>
</body>
</html>
        """
        
        # Save the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated successfully!")
        print(f"📄 Report saved as: {output_file}")
        print(f"🌐 Open the file in a web browser to view the report")
        
        return output_file

# Main execution function
def generate_safety_inspection_html_report(df, claude_analysis=None, report_title=None, output_file=None):
    """
    Generate comprehensive HTML report for safety inspection analysis
    
    Args:
        df: DataFrame with safety inspection data
        claude_analysis: Results from Claude analysis (optional)
        report_title: Custom title for the report
        output_file: Custom output filename
    
    Returns:
        Path to generated HTML file
    """
    
    print("🚀 Starting HTML Report Generation...")
    print("="*60)
    
    # Set default title if not provided
    if report_title is None:
        report_title = "Safety Inspection Analysis Report"
    
    # Initialize report generator
    report_generator = SafetyAnalysisHTMLReportGenerator(
        df=df,
        claude_analysis=claude_analysis,
        report_title=report_title
    )
    
    # Generate the report
    output_path = report_generator.generate_full_html_report(output_file)
    
    # Display summary
    print(f"\n📊 Report Summary:")
    print(f"   Title: {report_title}")
    print(f"   Records analyzed: {len(df):,}")
    print(f"   Report file: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path

# Example usage with sample data
def create_sample_report():
    """Create a sample report for demonstration"""
    
    # Create sample data structure
    sample_data = {
        'INSPECTION_DATE': pd.date_range('2025-01-01', periods=100, freq='D'),
        'AUTHOR_NAME': np.random.choice(['John Smith', 'Jane Doe', 'Mike Johnson'], 100),
        'SITE_NAME': np.random.choice(['Endurance', 'Platform A', 'Platform B'], 100),
        'FACILITY_AREA': np.random.choice(['Deck Storage', 'Main Deck', 'Drill Floor'], 100),
        'RESPONSE': np.random.choice(['Effective', 'Ineffective'], 100),
        'COMMENT': ['Sample inspection comment ' + str(i) for i in range(100)]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Sample Claude analysis
    sample_claude_analysis = """
    **COMMON THEMES AND PATTERNS:**
    
    • **Equipment Maintenance** (Found in 34% of comments)
      - "Equipment checks completed satisfactorily"
      - "Maintenance schedules being followed"
    
    • **Training and Competency** (Found in 28% of comments)  
      - "Personnel demonstrate adequate training"
      - "Certification requirements met"
    
    **POSITIVE OBSERVATIONS:**
    
    • **Safety Compliance** (23% of comments)
      - "All safety procedures followed correctly"
      - "PPE usage is excellent"
    
    **COMMON ISSUES:**
    
    • **Documentation Gaps** (15% of comments)
      - "Some paperwork incomplete"
      - "Records need updating"
    """
    
    # Generate report
    output_file = generate_safety_inspection_html_report(
        df=sample_df,
        claude_analysis=sample_claude_analysis,
        report_title="Sample Safety Inspection Report",
        output_file="sample_safety_report.html"
    )
    
    return output_file

# Usage example
if __name__ == "__main__":
    
    print("="*80)
    print("SAFETY INSPECTION HTML REPORT GENERATOR")
    print("="*80)
    
    try:
        # Option 1: Load your actual data
        # df = pd.read_csv("your_safety_inspection_data.csv")
        # claude_results = {"analysis": "Your Claude analysis results here"}
        # report_path = generate_safety_inspection_html_report(df, claude_results)
        
        # Option 2: Create sample report for demonstration
        print("📊 Creating sample report for demonstration...")
        sample_report = create_sample_report()
        
        print(f"\n✅ Sample report created: {sample_report}")
        print("🌐 Open the HTML file in your web browser to view the report")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
    
    print("\n" + "="*80)
    print("HTML REPORT FEATURES:")
    print("="*80)
    print("✅ Professional Bootstrap-based design")
    print("✅ Interactive Plotly charts and visualizations")
    print("✅ Responsive layout for all devices")
    print("✅ Executive summary with key metrics")
    print("✅ Data quality assessment with recommendations")
    print("✅ Response pattern analysis with trends")
    print("✅ AI-powered comment analysis integration") 
    print("✅ Site and facility breakdown analysis")
    print("✅ Timeline analysis with activity patterns")
    print("✅ Actionable recommendations and next steps")
    print("✅ Table of contents with smooth navigation")
    print("✅ Print-friendly styling")
    print("✅ Professional color scheme and typography")
    print("\n🚀 Perfect for executive presentations and stakeholder reporting!")
