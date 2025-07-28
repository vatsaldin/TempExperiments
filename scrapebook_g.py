import re
from datetime import datetime, timedelta
from enum import Enum

class ReportPeriod(Enum):
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"

class SafetyReportGenerator:
    def __init__(self):
        self.report_period = ReportPeriod.LAST_WEEK  # Default
        self.custom_start_date = None
        self.custom_end_date = None
    
    def set_report_period(self, period, start_date=None, end_date=None):
        """
        Set the reporting period for analysis
        
        Args:
            period (ReportPeriod): The time period for the report
            start_date (datetime, optional): Custom start date for CUSTOM period
            end_date (datetime, optional): Custom end date for CUSTOM period
        """
        self.report_period = period
        if period == ReportPeriod.CUSTOM:
            self.custom_start_date = start_date
            self.custom_end_date = end_date
    
    def get_date_range(self):
        """
        Calculate the date range based on the selected report period
        
        Returns:
            tuple: (start_date, end_date, period_description)
        """
        end_date = datetime.now()
        
        if self.report_period == ReportPeriod.LAST_DAY:
            start_date = end_date - timedelta(days=1)
            description = "Last 24 Hours"
            
        elif self.report_period == ReportPeriod.LAST_WEEK:
            start_date = end_date - timedelta(weeks=1)
            description = "Last 7 Days"
            
        elif self.report_period == ReportPeriod.LAST_MONTH:
            start_date = end_date - timedelta(days=30)
            description = "Last 30 Days"
            
        elif self.report_period == ReportPeriod.CUSTOM:
            start_date = self.custom_start_date or (end_date - timedelta(weeks=1))
            end_date = self.custom_end_date or end_date
            description = f"Custom Period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
        
        return start_date, end_date, description
    
    def filter_data_by_date(self, inspection_data, date_field='inspection_date'):
        """
        Filter inspection data based on the selected date range
        
        Args:
            inspection_data (list): List of inspection records
            date_field (str): Name of the date field in the data
        
        Returns:
            list: Filtered inspection data
        """
        start_date, end_date, _ = self.get_date_range()
        filtered_data = []
        
        for record in inspection_data:
            if date_field in record:
                try:
                    # Parse the date (adjust format as needed)
                    record_date = datetime.strptime(record[date_field], '%Y-%m-%d')
                    if start_date <= record_date <= end_date:
                        filtered_data.append(record)
                except ValueError:
                    # Handle different date formats or invalid dates
                    continue
        
        return filtered_data
    
    def generate_analysis_summary(self, analysis_text, inspection_count=0, incident_count=0):
        """
        Generate analysis summary with date-aware context
        
        Args:
            analysis_text (str): Raw analysis text
            inspection_count (int): Number of inspections in the period
            incident_count (int): Number of incidents in the period
        
        Returns:
            str: HTML formatted analysis with date context
        """
        start_date, end_date, period_description = self.get_date_range()
        
        # Extract sections from analysis
        positive_observations = self.extract_section(analysis_text, "POSITIVE OBSERVATIONS")
        improvement_opportunities = self.extract_section(analysis_text, "IMPROVEMENT OPPORTUNITIES")
        risk_assessment = self.extract_section(analysis_text, "RISK ASSESSMENT")
        key_recommendations = self.extract_section(analysis_text, "Key recommendations")
        
        # Generate HTML report with date context
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Safety Analysis Report - {period_description}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 25px; }}
        .section h2 {{ color: #2c5aa0; border-bottom: 2px solid #2c5aa0; padding-bottom: 5px; }}
        .stats {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .positive {{ color: #008000; }}
        .warning {{ color: #ff6600; }}
        .critical {{ color: #cc0000; }}
        ul {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Safety Analysis Report</h1>
        <p><strong>Report Period:</strong> {period_description}</p>
        <p><strong>Date Range:</strong> {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    
    <div class="stats">
        <h3>Period Summary</h3>
        <p>📊 <strong>Total Inspections:</strong> {inspection_count}</p>
        <p>⚠️ <strong>Total Incidents:</strong> {incident_count}</p>
        <p>📈 <strong>Inspection Rate:</strong> {inspection_count / max(1, (end_date - start_date).days):.1f} per day</p>
    </div>
    
    <div class="section">
        <h2>✅ Positive Observations</h2>
        <p>{positive_observations}</p>
    </div>
    
    <div class="section">
        <h2>🔧 Improvement Opportunities</h2>
        <p>{improvement_opportunities}</p>
    </div>
    
    <div class="section">
        <h2>⚡ Risk Assessment</h2>
        <p>{risk_assessment}</p>
    </div>
    
    <div class="section">
        <h2>📋 Key Recommendations</h2>
        <p>{key_recommendations}</p>
    </div>
    
    <div class="section">
        <h2>📊 Period-Specific Insights</h2>
        <p>{self.generate_period_insights(inspection_count, incident_count)}</p>
    </div>
</body>
</html>"""
        
        return html_content
    
    def generate_period_insights(self, inspection_count, incident_count):
        """
        Generate insights specific to the reporting period
        """
        start_date, end_date, period_description = self.get_date_range()
        days_in_period = (end_date - start_date).days
        
        insights = []
        
        if self.report_period == ReportPeriod.LAST_DAY:
            insights.append(f"📅 Daily snapshot analysis for {end_date.strftime('%B %d, %Y')}")
            if inspection_count > 0:
                insights.append(f"✅ {inspection_count} inspection(s) completed in the last 24 hours")
            else:
                insights.append("⚠️ No inspections recorded in the last 24 hours")
                
        elif self.report_period == ReportPeriod.LAST_WEEK:
            avg_daily = inspection_count / 7
            insights.append(f"📊 Weekly analysis showing {avg_daily:.1f} average inspections per day")
            if avg_daily < 1:
                insights.append("🔴 Below recommended daily inspection frequency")
            else:
                insights.append("✅ Meeting daily inspection targets")
                
        elif self.report_period == ReportPeriod.LAST_MONTH:
            avg_weekly = inspection_count / 4.3
            insights.append(f"📈 Monthly analysis showing {avg_weekly:.1f} average inspections per week")
            
            # Incident rate analysis
            if incident_count > 0:
                incident_rate = (incident_count / inspection_count) * 100 if inspection_count > 0 else 0
                insights.append(f"⚠️ Incident rate: {incident_rate:.1f}% of inspections")
            else:
                insights.append("✅ Zero incidents reported during this period")
        
        # Add seasonal or temporal recommendations
        current_month = end_date.month
        if current_month in [12, 1, 2]:  # Winter months
            insights.append("❄️ Winter period: Increased focus on slip/fall hazards and cold weather PPE")
        elif current_month in [6, 7, 8]:  # Summer months
            insights.append("☀️ Summer period: Heightened attention to heat stress and hydration protocols")
        
        return "<br>".join([f"• {insight}" for insight in insights])
    
    def extract_section(self, analysis_text, section_name):
        """
        Enhanced utility to extract a section from the analysis text using section headers.
        Returns the section as HTML (preserves line breaks and formatting).
        """
        import re
        
        # Try multiple patterns to find the section
        patterns = [
            rf"{re.escape(section_name)}\s*:?\s*\n(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
            rf"^{re.escape(section_name)}\s*:?\s*$(.*?)(?=^[A-Z][A-Z\s]+:|$)",
            rf"{re.escape(section_name)}[:\-]*\s*(.*?)(?=\n\n|\n[A-Z]|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                section = match.group(1).strip()
                
                # Convert bullet points and line breaks to HTML
                section = re.sub(r'\n\s*[-•]\s*', '\n• ', section)
                section = section.replace('\n• ', '<br>• ')
                section = section.replace('\n', '<br>')
                
                return section
        
        return f"<i>No data found for {section_name} section.</i>"
    
    def save_report(self, html_content, base_filename="safety_report"):
        """
        Save the HTML report with a date-specific filename
        """
        start_date, end_date, _ = self.get_date_range()
        
        # Create filename with date range
        date_suffix = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
        filename = f"{base_filename}_{self.report_period.value}_{date_suffix}.html"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"📄 Safety report saved: {filename}")
        return filename

# Usage Example
def main():
    # Initialize the report generator
    generator = SafetyReportGenerator()
    
    # Example: Generate report for last week (default)
    generator.set_report_period(ReportPeriod.LAST_WEEK)
    
    # Example: Generate report for last day
    # generator.set_report_period(ReportPeriod.LAST_DAY)
    
    # Example: Generate report for last month
    # generator.set_report_period(ReportPeriod.LAST_MONTH)
    
    # Example: Generate custom date range report
    # custom_start = datetime(2024, 1, 1)
    # custom_end = datetime(2024, 1, 31)
    # generator.set_report_period(ReportPeriod.CUSTOM, custom_start, custom_end)
    
    # Sample analysis text (replace with your actual data)
    analysis_text = """
    POSITIVE OBSERVATIONS:
    - Effective use of PTW systems
    - Adherence to confined space entry procedures
    - Proper PPE usage
    - Regular equipment inspections and maintenance
    
    IMPROVEMENT OPPORTUNITIES:
    - Enhance barrier management and restricted zone access control
    - Improve documentation management (permits, certifications, safety data sheets)
    - Strengthen dropped object prevention and mitigation measures
    
    RISK ASSESSMENT:
    The analysis of safety inspection comments revealed several key themes, including high-risk categories and job planning considerations for workplace safety protocols.
    
    Key recommendations:
    1. Implement comprehensive barrier management program
    2. Establish robust documentation management system
    3. Strengthen dropped object prevention strategies
    """
    
    # Generate the report
    html_report = generator.generate_analysis_summary(
        analysis_text, 
        inspection_count=45,  # Example data
        incident_count=2      # Example data
    )
    
    # Save the report
    filename = generator.save_report(html_report)
    
    print(f"Report generated for period: {generator.get_date_range()[2]}")

if __name__ == "__main__":
    main()
