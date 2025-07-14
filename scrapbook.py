# Claude Prompt for Safety Inspection Comment Analysis

def generate_claude_prompt_for_comment_analysis(df, sample_size=100):
    """
    Generate a comprehensive prompt for Claude to analyze safety inspection comments
    
    Args:
        df: DataFrame with safety inspection data
        sample_size: Number of sample comments to include in the prompt
    
    Returns:
        Formatted prompt string for Claude analysis
    """
    
    # Get sample comments (mix of different responses if available)
    if 'RESPONSE' in df.columns:
        # Get balanced sample from different response types
        sample_comments = []
        for response_type in df['RESPONSE'].unique():
            if pd.notna(response_type):
                subset = df[df['RESPONSE'] == response_type]['COMMENT'].dropna()
                sample_size_per_type = min(20, len(subset))
                if sample_size_per_type > 0:
                    sample_subset = subset.sample(n=sample_size_per_type, random_state=42)
                    for idx, comment in sample_subset.items():
                        sample_comments.append({
                            'response_type': response_type,
                            'comment': str(comment)[:500],  # Limit length
                            'site': df.loc[idx, 'SITE_NAME'] if 'SITE_NAME' in df.columns else 'Unknown',
                            'facility': df.loc[idx, 'FACILITY_AREA'] if 'FACILITY_AREA' in df.columns else 'Unknown'
                        })
    else:
        # Random sample if no response column
        sample_subset = df['COMMENT'].dropna().sample(n=min(sample_size, len(df)), random_state=42)
        sample_comments = []
        for idx, comment in sample_subset.items():
            sample_comments.append({
                'response_type': 'Unknown',
                'comment': str(comment)[:500],
                'site': df.loc[idx, 'SITE_NAME'] if 'SITE_NAME' in df.columns else 'Unknown',
                'facility': df.loc[idx, 'FACILITY_AREA'] if 'FACILITY_AREA' in df.columns else 'Unknown'
            })
    
    # Generate the prompt
    prompt = f"""
You are an expert safety analyst reviewing inspection comments from offshore drilling operations. Please analyze the following safety inspection comments to identify patterns, themes, and insights.

DATASET CONTEXT:
- Total Comments: {len(df['COMMENT'].dropna())}
- Date Range: {df['INSPECTION_DATE'].min()} to {df['INSPECTION_DATE'].max()}
- Sites: {df['SITE_NAME'].nunique() if 'SITE_NAME' in df.columns else 'Unknown'}
- Facility Areas: {df['FACILITY_AREA'].nunique() if 'FACILITY_AREA' in df.columns else 'Unknown'}
- Response Types: {list(df['RESPONSE'].unique()) if 'RESPONSE' in df.columns else 'Unknown'}

SAMPLE COMMENTS FOR ANALYSIS:
"""
    
    # Add sample comments to prompt
    for i, comment_data in enumerate(sample_comments[:100], 1):  # Limit to 100 for prompt length
        prompt += f"""
{i}. Response: {comment_data['response_type']} | Site: {comment_data['site']} | Area: {comment_data['facility']}
   Comment: "{comment_data['comment']}"
"""
    
    prompt += f"""

ANALYSIS REQUESTED:

1. COMMON THEMES AND PATTERNS:
   - Identify the 5-7 most frequent themes across all comments
   - For each theme, provide:
     * Theme name and description
     * Frequency/prevalence 
     * Exact quote examples from the comments above
     * Which sites/areas this theme appears most in

2. POSITIVE OBSERVATIONS:
   - Identify positive safety practices mentioned
   - Good examples of compliance and safety measures
   - Effective procedures being followed
   - Provide exact quotes showing these positive observations
   - Quantify how often positive observations occur

3. COMMON ISSUES AND CONCERNS:
   - Most frequently mentioned safety issues
   - Recurring problems across different sites/areas
   - Equipment or procedural concerns
   - Provide exact quotes highlighting these issues
   - Assess severity and frequency of issues

4. OPPORTUNITIES FOR IMPROVEMENT (OFI):
   - Areas where improvements are suggested
   - Recurring suggestions for better practices
   - Equipment or training needs mentioned
   - Provide exact quotes showing improvement opportunities

5. SITE-SPECIFIC PATTERNS:
   - Are there patterns specific to certain sites?
   - Do different facility areas have different types of issues?
   - Provide examples with exact quotes

6. RESPONSE TYPE ANALYSIS:
   - How do comments differ between "Effective" vs "Ineffective" responses?
   - What characterizes effective vs ineffective observations?
   - Provide comparative examples with exact quotes

7. RISK CATEGORIES:
   - Categorize the safety risks mentioned (e.g., equipment, procedural, environmental)
   - Rank by frequency and potential severity
   - Provide examples for each category

8. EXECUTIVE SUMMARY:
   - Overall safety culture assessment based on comments
   - Top 3 strengths identified
   - Top 3 areas needing attention
   - Key recommendations for management

FORMATTING REQUIREMENTS:
- Use clear headings and bullet points
- Include exact quotes in quotation marks with the comment number reference
- Provide statistics where possible (e.g., "appears in 23% of comments")
- Bold key findings
- End with actionable recommendations

Please be thorough and specific in your analysis, ensuring all findings are supported by exact examples from the provided comments.
"""
    
    return prompt

# Usage function
def analyze_comments_with_claude(df, claude_client=None):
    """
    Analyze safety inspection comments using Claude
    
    Args:
        df: DataFrame with inspection data
        claude_client: Initialized Claude client (optional)
    
    Returns:
        Analysis results from Claude
    """
    
    print("🔍 Generating Claude analysis prompt for safety inspection comments...")
    
    # Generate the prompt
    prompt = generate_claude_prompt_for_comment_analysis(df)
    
    # Save prompt to file for review
    prompt_file = "claude_analysis_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"📄 Prompt saved to: {prompt_file}")
    print(f"📊 Prompt includes {len(df['COMMENT'].dropna())} total comments")
    print(f"📝 Sample size in prompt: {min(100, len(df['COMMENT'].dropna()))}")
    
    # If Claude client is available, make the API call
    if claude_client:
        try:
            print("🤖 Sending request to Claude...")
            
            # Make API call to Claude
            response = claude_client.get_claude_insights(prompt)
            
            # Save response to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_file = f"claude_comment_analysis_{timestamp}.txt"
            
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("CLAUDE SAFETY INSPECTION COMMENT ANALYSIS\n")
                f.write("="*80 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {len(df)} total records\n")
                f.write("="*80 + "\n\n")
                f.write(response)
            
            print(f"✅ Claude analysis completed!")
            print(f"📄 Results saved to: {response_file}")
            
            return response
            
        except Exception as e:
            print(f"❌ Error calling Claude API: {e}")
            print("📄 You can copy the prompt from claude_analysis_prompt.txt and paste it manually into Claude")
            
    else:
        print("ℹ️  No Claude client provided.")
        print("📋 Next steps:")
        print("   1. Copy the content from 'claude_analysis_prompt.txt'")
        print("   2. Paste it into Claude (web interface or API)")
        print("   3. Review the detailed analysis results")
    
    return prompt

# Alternative: Simple prompt generator for manual use
def generate_simple_claude_prompt(comments_list, max_comments=50):
    """
    Generate a simpler prompt for manual Claude analysis
    
    Args:
        comments_list: List of comment strings
        max_comments: Maximum number of comments to include
    
    Returns:
        Simple prompt string
    """
    
    sample_comments = comments_list[:max_comments]
    
    prompt = f"""
Analyze these {len(sample_comments)} safety inspection comments from offshore drilling operations. Please identify:

1. **Common Themes** (with exact quote examples)
2. **Positive Observations** (good practices mentioned)
3. **Common Issues** (recurring problems)
4. **Improvement Opportunities** (suggestions for better practices)
5. **Executive Summary** (key insights and recommendations)

COMMENTS TO ANALYZE:
"""
    
    for i, comment in enumerate(sample_comments, 1):
        prompt += f"\n{i}. \"{str(comment)[:300]}...\"\n"
    
    prompt += """
Please provide specific examples with exact quotes for each category identified.
"""
    
    return prompt

# Example usage with your data
if __name__ == "__main__":
    
    print("="*80)
    print("CLAUDE COMMENT ANALYSIS PROMPT GENERATOR")
    print("="*80)
    
    # Example usage (replace with your actual data loading)
    try:
        # Load your data
        df = pd.read_csv("your_safety_inspection_data.csv")
        
        # Method 1: Full comprehensive analysis
        print("\n🔍 Method 1: Comprehensive Analysis")
        prompt = analyze_comments_with_claude(df)
        
        # Method 2: Simple analysis for manual use
        print("\n📝 Method 2: Simple Prompt for Manual Use")
        comments_list = df['COMMENT'].dropna().tolist()
        simple_prompt = generate_simple_claude_prompt(comments_list, max_comments=30)
        
        with open("simple_claude_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(simple_prompt)
        
        print("📄 Simple prompt saved to: simple_claude_prompt.txt")
        
    except FileNotFoundError:
        print("❌ Please update the file path with your actual CSV file")
        print("\n📋 Usage Instructions:")
        print("1. Load your DataFrame: df = pd.read_csv('your_file.csv')")
        print("2. Generate prompt: prompt = analyze_comments_with_claude(df)")
        print("3. Copy prompt from saved file and paste into Claude")
        print("4. Get comprehensive analysis results")
    
    print("\n" + "="*80)
    print("WHAT THE CLAUDE ANALYSIS WILL PROVIDE:")
    print("="*80)
    print("✅ Common themes with exact quote examples")
    print("✅ Positive safety observations identified")
    print("✅ Recurring issues and concerns")
    print("✅ Opportunities for improvement")
    print("✅ Site-specific patterns")
    print("✅ Response type analysis (Effective vs Ineffective)")
    print("✅ Risk categorization and ranking")
    print("✅ Executive summary with actionable recommendations")
    print("✅ All findings supported by exact text examples")
    print("\n🚀 Perfect for safety management reporting!")
