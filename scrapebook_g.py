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
import time
import json
import os
from datetime import datetime
import warnings
import boto3
from botocore.exceptions import ReadTimeoutError, ClientError, EndpointConnectionError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class IntegratedSafetyAnalysisSystem:
    """
    Complete integrated system for safety inspection analysis with:
    - Robust Claude API handling with timeout management
    - Chunked comment analysis for large datasets
    - Professional HTML report generation
    - Interactive visualizations and insights
    """
    
    def __init__(self, region_name="us-east-1", max_retries=3, timeout_seconds=300):
        """
        Initialize the integrated analysis system
        
        Args:
            region_name: AWS region for Bedrock
            max_retries: Maximum retry attempts for API calls
            timeout_seconds: Timeout for API calls in seconds
        """
        self.region_name = region_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.client_available = False
        
        # Initialize Bedrock client
        self.initialize_bedrock_client()
        
        # Analysis configuration
        self.analysis_config = {
            'max_tokens_per_request': 4000,
            'chunk_size': 30,  # Comments per chunk
            'retry_delay': 5,  # Seconds between retries
            'fallback_enabled': True
        }
        
        # Report styling
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
        
        # Results storage
        self.df = None
        self.claude_analysis = None
        self.analysis_results = {}
        self.partial_results = []
        self.chart_counter = 1
    
    def initialize_bedrock_client(self):
        """Initialize Bedrock client with timeout configuration"""
        try:
            config = boto3.session.Config(
                region_name=self.region_name,
                retries={'max_attempts': self.max_retries},
                read_timeout=self.timeout_seconds,
                connect_timeout=30
            )
            
            self.bedrock_runtime = boto3.client('bedrock-runtime', config=config)
            self.test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            self.client_available = False
    
    def test_connection(self):
        """Test connection to Bedrock Claude"""
        try:
            test_prompt = "Hello, please respond with 'Connection successful'"
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": test_prompt}]
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and len(response_body['content']) > 0:
                logger.info("✅ Claude connection test successful")
                self.client_available = True
            else:
                self.client_available = False
                
        except Exception as e:
            logger.error(f"❌ Claude connection test failed: {e}")
            self.client_available = False
    
    def make_robust_api_call(self, prompt, max_tokens=4000):
        """Make robust API call with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 API call attempt {attempt + 1}/{self.max_retries}")
                
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "top_k": 250,
                    "top_p": 0.999
                })
                
                start_time = time.time()
                
                response = self.bedrock_runtime.invoke_model(
                    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                end_time = time.time()
                logger.info(f"⏱️ API call completed in {end_time - start_time:.2f} seconds")
                
                response_body = json.loads(response['body'].read())
                
                if 'content' in response_body and len(response_body['content']) > 0:
                    return response_body['content'][0]['text']
                else:
                    logger.warning("⚠️ Empty response from Claude")
                    return None
                    
            except ReadTimeoutError as e:
                logger.warning(f"⏰ Timeout error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.analysis_config['retry_delay'] * (attempt + 1)
                    logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("❌ All retry attempts failed due to timeout")
                    return None
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.error(f"🚨 AWS Client error: {error_code} - {e}")
                
                if error_code in ['ThrottlingException', 'ServiceUnavailable']:
                    if attempt < self.max_retries - 1:
                        wait_time = self.analysis_config['retry_delay'] * (2 ** attempt)
                        logger.info(f"⏳ Exponential backoff: waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                return None
                
            except Exception as e:
                logger.error(f"🚨 Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.analysis_config['retry_delay'])
                    continue
                else:
                    return None
        
        return None
    
    def load_data(self, file_path):
        """Load and validate safety inspection data"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
            
            # Convert INSPECTION_DATE to datetime
            if 'INSPECTION_DATE' in self.df.columns:
                self.df['INSPECTION_DATE'] = pd.to_datetime(self.df['INSPECTION_DATE'], errors='coerce')
                print(f"📅 Date range: {self.df['INSPECTION_DATE'].min()} to {self.df['INSPECTION_DATE'].max()}")
            
            # Validate expected columns
            expected_cols = ['INSPECTION_DATE', 'AUTHOR_NAME', 'PREPARED_BY', 'SITE_NAME', 
                           'FACILITY_AREA', 'TEMPLATE_NAME', 'QUESTION_LABEL', 'RESPONSE', 'COMMENT']
            
            missing_cols = [col for col in expected_cols if col not in self.df.columns]
            if missing_cols:
                print(f"⚠️ Missing expected columns: {missing_cols}")
            
            available_cols = [col for col in expected_cols if col in self.df.columns]
            print(f"✅ Available columns: {available_cols}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_chunk_prompt(self, chunk_df, chunk_num, total_chunks):
        """Create analysis prompt for a specific chunk"""
        sample_comments = []
        for idx, row in chunk_df.iterrows():
            comment_data = {
                'comment': str(row['COMMENT'])[:400],
                'response': row.get('RESPONSE', 'Unknown'),
                'site': row.get('SITE_NAME', 'Unknown'),
                'facility': row.get('FACILITY_AREA', 'Unknown'),
                'date': row.get('INSPECTION_DATE', 'Unknown')
            }
            sample_comments.append(comment_data)
        
        prompt = f"""
You are analyzing safety inspection comments from offshore drilling operations.

CHUNK INFORMATION:
- Chunk {chunk_num} of {total_chunks}
- Comments in this chunk: {len(chunk_df)}
- Analysis focus: Extract key themes, issues, and observations

COMMENTS TO ANALYZE:
"""
        
        for i, comment_data in enumerate(sample_comments, 1):
            prompt += f"""
{i}. Response: {comment_data['response']} | Site: {comment_data['site']} | Area: {comment_data['facility']}
   Comment: "{comment_data['comment']}"
"""
        
        prompt += f"""

ANALYSIS REQUEST (for this chunk only):
1. **Key Themes** - Identify 3-5 main themes in these comments
2. **Safety Issues** - List specific safety concerns mentioned
3. **Positive Observations** - Note good practices or compliance
4. **Common Keywords** - Extract frequently mentioned terms
5. **Risk Categories** - Categorize the types of risks mentioned

RESPONSE FORMAT:
- Be concise and specific
- Use bullet points
- Include brief quote examples
- Focus on actionable insights

Please analyze this chunk and provide insights that can be combined with other chunks.
"""
        
        return prompt
    
    def analyze_comments_in_chunks(self, chunk_size=None):
        """Analyze comments in chunks to avoid timeouts"""
        if chunk_size is None:
            chunk_size = self.analysis_config['chunk_size']
        
        print(f"🔄 Processing comments in chunks of {chunk_size}...")
        
        # Get clean comments
        comments_df = self.df[self.df['COMMENT'].notna() & (self.df['COMMENT'] != '')].copy()
        total_comments = len(comments_df)
        
        print(f"📊 Total comments to analyze: {total_comments}")
        
        if total_comments == 0:
            print("❌ No comments found for analysis")
            return None
        
        # Split into chunks
        chunks = []
        for i in range(0, len(comments_df), chunk_size):
            chunk = comments_df.iloc[i:i+chunk_size]
            chunks.append(chunk)
        
        print(f"📦 Created {len(chunks)} chunks")
        
        # Analyze each chunk
        chunk_results = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n🔍 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} comments)...")
            
            # Create chunk-specific prompt
            chunk_prompt = self.create_chunk_prompt(chunk, i+1, len(chunks))
            
            # Make API call for this chunk
            result = self.make_robust_api_call(chunk_prompt, max_tokens=3000)
            
            if result:
                chunk_results.append({
                    'chunk_id': i+1,
                    'comment_count': len(chunk),
                    'analysis': result,
                    'status': 'success'
                })
                print(f"✅ Chunk {i+1} analysis completed")
                
                # Save partial result
                self.save_partial_result(chunk_results[-1], i+1)
                
            else:
                print(f"❌ Chunk {i+1} analysis failed")
                chunk_results.append({
                    'chunk_id': i+1,
                    'comment_count': len(chunk),
                    'analysis': None,
                    'status': 'failed'
                })
            
            # Brief pause between chunks
            if i < len(chunks) - 1:
                time.sleep(2)
        
        # Combine results
        return self.combine_chunk_results(chunk_results, total_comments)
    
    def save_partial_result(self, result, chunk_id):
        """Save partial result to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chunk_analysis_{chunk_id}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"CHUNK {chunk_id} ANALYSIS RESULT\n")
            f.write("="*50 + "\n")
            f.write(f"Comments analyzed: {result['comment_count']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            
            if result['analysis']:
                f.write(result['analysis'])
            else:
                f.write("Analysis failed for this chunk.")
        
        print(f"💾 Partial result saved: {filename}")
    
    def combine_chunk_results(self, chunk_results, total_comments):
        """Combine analysis results from all chunks"""
        print(f"\n🔄 Combining results from {len(chunk_results)} chunks...")
        
        successful_chunks = [r for r in chunk_results if r['status'] == 'success']
        failed_chunks = [r for r in chunk_results if r['status'] == 'failed']
        
        print(f"✅ Successful chunks: {len(successful_chunks)}")
        print(f"❌ Failed chunks: {len(failed_chunks)}")
        
        if not successful_chunks:
            print("❌ No successful chunk analysis. Creating fallback analysis...")
            return self.create_fallback_analysis(total_comments)
        
        # Create summary prompt
        summary_prompt = f"""
You are combining analysis results from {len(successful_chunks)} chunks of safety inspection comments.

CHUNK RESULTS TO COMBINE:
"""
        
        for i, result in enumerate(successful_chunks, 1):
            summary_prompt += f"""
CHUNK {result['chunk_id']} ({result['comment_count']} comments):
{result['analysis']}

---
"""
        
        summary_prompt += f"""

FINAL SYNTHESIS REQUEST:
Please combine these chunk analyses into a comprehensive summary with:

1. **OVERALL THEMES** - Top 5-7 themes across all chunks
2. **COMMON SAFETY ISSUES** - Most frequently mentioned problems
3. **POSITIVE OBSERVATIONS** - Good practices identified
4. **IMPROVEMENT OPPORTUNITIES** - Areas needing attention
5. **RISK ASSESSMENT** - Overall risk categories and severity
6. **EXECUTIVE SUMMARY** - Key findings and recommendations

Total comments analyzed: {sum(r['comment_count'] for r in successful_chunks)}
Success rate: {len(successful_chunks)}/{len(chunk_results)} chunks

Please provide a cohesive, actionable analysis suitable for executive reporting.
"""
        
        # Make final API call to combine results
        print("🔄 Making final API call to combine chunk results...")
        combined_result = self.make_robust_api_call(summary_prompt, max_tokens=4000)
        
        if combined_result:
            final_result = {
                'total_comments': total_comments,
                'successful_chunks': len(successful_chunks),
                'failed_chunks': len(failed_chunks),
                'analysis': combined_result,
                'chunk_details': chunk_results
            }
            
            self.claude_analysis = final_result
            self.save_final_analysis(final_result)
            return final_result
        else:
            print("❌ Final combination failed. Using best available chunk result...")
            return self.create_best_available_analysis(successful_chunks, total_comments)
    
    def create_fallback_analysis(self, total_comments):
        """Create fallback analysis when API calls fail"""
        fallback_content = f"""
**FALLBACK ANALYSIS REPORT**

Due to API connectivity issues, automated analysis could not be completed.
However, here are recommended manual analysis steps:

**MANUAL ANALYSIS RECOMMENDATIONS:**

1. **Text Mining Approach**:
   - Search for common keywords: "equipment", "training", "procedure", "safety"
   - Count frequency of terms like "effective", "ineffective", "issue", "concern"

2. **Categorization Strategy**:
   - Group comments by response type (Effective vs Ineffective)
   - Sort by site and facility area
   - Look for patterns in dates/timeframes

3. **Key Areas to Investigate**:
   - Equipment-related comments
   - Training and competency issues
   - Procedural compliance
   - Communication effectiveness

4. **Statistical Analysis**:
   - Calculate percentage of positive vs negative comments
   - Identify most active sites/facilities
   - Track trends over time

**NEXT STEPS:**
- Review individual comment files saved during processing
- Consider reducing batch size and retrying
- Use alternative analysis tools if needed

Total comments requiring analysis: {total_comments}
"""
        
        return {
            'total_comments': total_comments,
            'analysis': fallback_content,
            'status': 'fallback',
            'timestamp': datetime.now()
        }
    
    def save_final_analysis(self, result):
        """Save final combined analysis to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"safety_comment_analysis_final_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLAUDE SAFETY INSPECTION COMMENT ANALYSIS - FINAL REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total comments analyzed: {result['total_comments']}\n")
            f.write(f"Successful chunks: {result['successful_chunks']}\n")
            f.write(f"Failed chunks: {result['failed_chunks']}\n")
            f.write(f"Success rate: {(result['successful_chunks']/(result['successful_chunks']+result['failed_chunks']))*100:.1f}%\n")
            f.write("="*80 + "\n\n")
            f.write(result['analysis'])
        
        print(f"📄 Final analysis saved: {filename}")
    
    def create_best_available_analysis(self, successful_chunks, total_comments):
        """Create analysis from best available chunk when combination fails"""
        if not successful_chunks:
            return self.create_fallback_analysis(total_comments)
        
        best_chunk = max(successful_chunks, key=lambda x: x['comment_count'])
        
        analysis_content = f"""
**PARTIAL ANALYSIS REPORT (BEST AVAILABLE)**

Note: This analysis is based on the most complete chunk due to API limitations.

**CHUNK DETAILS:**
- Chunk ID: {best_chunk['chunk_id']}
- Comments analyzed: {best_chunk['comment_count']} out of {total_comments} total
- Representation: {(best_chunk['comment_count']/total_comments)*100:.1f}% of total comments

**ANALYSIS RESULTS:**
{best_chunk['analysis']}

**LIMITATIONS:**
- This represents only a portion of your complete dataset
- For full analysis, consider running smaller batches
- Some patterns may not be visible in this subset

**RECOMMENDATIONS:**
- Review all chunk files for additional insights
- Consider manual review of remaining comments
- Retry analysis with smaller chunk sizes if needed
"""
        
        return {
            'total_comments': total_comments,
            'analyzed_comments': best_chunk['comment_count'],
            'analysis': analysis_content,
            'status': 'partial',
            'timestamp': datetime.now()
        }
    
    # HTML Report Generation Methods
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
        """Create executive summary section for HTML report"""
        total_records = len(self.df)
        date_range = f"{self.df['INSPECTION_DATE'].min().strftime('%Y-%m-%d')} to {self.df['INSPECTION_DATE'].max().strftime('%Y-%m-%d')}"
        unique_sites = self.df['SITE_NAME'].nunique() if 'SITE_NAME' in self.df.columns else 0
        unique_facilities = self.df['FACILITY_AREA'].nunique() if 'FACILITY_AREA' in self.df.columns else 0
        
        # Response distribution
        if 'RESPONSE' in self.df.columns:
            effective_count = self.df[self.df['RESPONSE'] == 'Effective'].shape[0]
            effectiveness_rate = (effective_count / total_records) * 100
        else:
            effectiveness_rate = 0
        
        comment_availability = (self.df['COMMENT'].notna().sum() / total_records) * 100
        
        # AI analysis status
        ai_status = "✅ Completed" if self.claude_analysis and 'analysis' in self.claude_analysis else "❌ Not Available"
        
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
                                    <li><strong>AI Analysis Status:</strong> {ai_status}</li>
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
                                            <h4 class="text-info">{len(self.df['RESPONSE'].unique()) if 'RESPONSE' in self.df.columns else 0}</h4>
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
    
    def create_claude_analysis_section(self):
        """Create Claude analysis results section for HTML report"""
        if not self.claude_analysis or 'analysis' not in self.claude_analysis:
            return f"""
            <div class="row mb-4">
                <div class="col-12">
                    <div class="alert alert-warning">
                        <h5 class="alert-heading">
                            <i class="fas fa-robot"></i> AI Analysis Status
                        </h5>
                        <p>Claude AI analysis was not available or failed to complete. This could be due to:</p>
                        <ul>
                            <li>API timeout or connectivity issues</li>
                            <li>Large dataset requiring chunked processing</li>
                            <li>AWS Bedrock access configuration</li>
                        </ul>
                        <hr>
                        <p class="mb-0">The report includes comprehensive statistical analysis. Consider running the analysis again with smaller chunk sizes for AI insights.</p>
                    </div>
                </div>
            </div>
            """
        
        # Format Claude analysis content
        analysis_content = self.claude_analysis['analysis']
        formatted_html = self.format_claude_analysis(analysis_content)
        
        # Add analysis metadata
        metadata_html = ""
        if 'successful_chunks' in self.claude_analysis:
            total_chunks = self.claude_analysis['successful_chunks'] + self.claude_analysis['failed_chunks']
            success_rate = (self.claude_analysis['successful_chunks'] / total_chunks) * 100
            
            metadata_html = f"""
            <div class="alert alert-info mb-3">
                <h6 class="alert-heading">Analysis Metadata</h6>
                <div class="row">
                    <div class="col-md-3">
                        <strong>Comments Analyzed:</strong><br>
                        {self.claude_analysis['total_comments']:,}
                    </div>
                    <div class="col-md-3">
                        <strong>Processing Chunks:</strong><br>
                        {self.claude_analysis['successful_chunks']}/{total_chunks}
                    </div>
                    <div class="col-md-3">
                        <strong>Success Rate:</strong><br>
                        {success_rate:.1f}%
                    </div>
                    <div class="col-md-3">
                        <strong>Analysis Status:</strong><br>
                        {'Complete' if success_rate > 80 else 'Partial'}
                    </div>
                </div>
            </div>
            """
        
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
                        {metadata_html}
                        {formatted_html}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return claude_section
    
    def format_claude_analysis(self, analysis_content):
        """Format Claude analysis content into structured HTML"""
        lines = analysis_content.split('\n')
        formatted_html = ""
        in_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for headers
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
                
                if '"' in list_item:
                    formatted_html += f'<li class="list-group-item"><i class="fas fa-quote-left text-muted"></i> {list_item}</li>\n'
                else:
                    formatted_html += f'<li class="list-group-item">{list_item}</li>\n'
                    
            else:
                if in_list:
                    formatted_html += "</ul>\n"
                    in_list = False
                
                if '"' in line:
                    formatted_html += f'<div class="alert alert-light border-left-primary"><i class="fas fa-quote-left text-muted"></i> {line}</div>\n'
                else:
                    formatted_html += f'<p class="mb-2">{line}</p>\n'
        
        if in_list:
            formatted_html += "</ul>\n"
        
        return formatted_html
    
    def create_response_analysis_section(self):
        """Create response pattern analysis section"""
        if 'RESPONSE' not in self.df.columns:
            return '<div class="alert alert-warning">Response column not available for analysis.</div>'
        
        response_counts = self.df['RESPONSE'].value_counts()
        
        # Create interactive charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Distribution', 'Response by Site', 
                           'Response Timeline', 'Response by Facility Area'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
