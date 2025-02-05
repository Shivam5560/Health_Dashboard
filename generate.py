import ollama
import json
import time

def generate_health_insights(input_data):
    """Improved version with better prompt engineering and parsing"""
    # Structured prompt with explicit formatting instructions
    prompt = f"""
    **Comprehensive Health Risk Assessment & Clinical Guidance**

    Analyze the provided patient data and generate a structured JSON response with specific, clinically-relevant recommendations. Maintain formal medical terminology while ensuring actionable directives and have 3-4 suggestions for each values of the subkeys and keep the tone formal and with no repetitive action verbs.
    Based on risk_scores parameters having scores of ascvd,chd,ckd,diabetes,stroke,overall risk percentage metrics and many more, have the output of insights and action items and recommendation appended with some from this as well for all 3 keys, so we would have a balanced and precise keys values, repetition of same things can be skipped/changed from every other values.
    
    Patient Data:
    {json.dumps(input_data, indent=2)}

    Generate JSON response taking help of risk_scores in patient data along with :

    1. **Recommendations**: 
    - Lifestyle: Specific interventions targeting identified risk factors
    - Monitoring: Concrete measurement protocols and frequency
    - Medication: Potential pharmacological options with rationale

    2. **Insights**:
    - all this to be done with validating numbers as well and comparing the results for different records(latest age - previous age)
    - Primary risks: Quantified pathological factors with severity assessment
    - Trends: Temporal analysis of critical biomarkers
    - Historical context: Clinical progression compared to previous assessments
    

    3. **Actions**:
    - Immediate: Time-sensitive interventions (<72h)
    - Long-term: 3-6 month objectives with success metrics
    - Follow-up: Specific timeline and assessment targets

    4. **Summary**: Concise clinical synopsis (120 words) prioritizing acute risks and critical interventions.
    
    5. **extra_score** : This returns the risk points for each parameter along with the corresponding score category, which can be 'low', 'moderate', or 'high'. The score cannot be 0, and the category depends on the score or points for each metric (e.g., the hypertension score category is determined by the hypertension score, and the same applies to the other three metrics)."
    
    
    Structure response exactly as given below json format no changes in it:
    ```json
    {{
        "recommendations": {{
            "lifestyle": ["item1", "item2",..],
            "monitoring": ["item1", "item2",..],
            "medication": ["Consider discussing medication options with a healthcare provider, including: item1",..]
        }},
        "insights": {{
            "primary_risk_factors": ["factor1", "factor2",..],
            "trend_analysis": "Detailed analysis...",
            "historical_comparisons": ["comparison1", "comparison2",...]
        }},
        "action_items": {{
            "immediate_actions": ["action1", "action2",..],
            "long_term_goals": ["goal1", "goal2",..],
            "follow_up_schedule": ["Schedule details",..]
        }},
        "Summary": "",

        "extra_score":{{
            "hypertension_score":,
            "depression_score":,
            "anxiety_score":,
            "stress_score":,
            "hypertension_score_category":,
            "depression_score_category":,
            "anxiety_score_category":,
            "stress_score_category":,
        }}
    }}
    **Composition Guidelines:**
    1. **Terminology & Precision:**
    - Use active verbs: Implement, Achieve, Monitor, Consider
    - Include numerical thresholds also for risk_scores of cvd and all (e.g., BP >140/90 mmHg)
    - Reference clinical standards (JNC, ADA, ACC/AHA)

    2. **Temporal Specificity:**
    - Define clear timeframes: "Within 72h", "Every 6-8 weeks"
    - Quantify trends: "12% reduction over 6 months"
    - check for risk_scores key and have the output of insights and action items ad recommendation from this also. 

    4. **Intervention Design:**
    - Pair actions with metrics
    - Specify monitoring modalities

    5. **Differentiation:**
    - Avoid overlapping recommendations across categories
    - Ensure medication suggestions match risk severity

    **Validation Requirements:**
    - Cross-validate biomarker trends with historical data
    - Align medication suggestions with current BMI/labs
    - Confirm temporal consistency in follow-up scheduling
    

    Generate a response that enables immediate clinical action while establishing measurable health targets.
    """

    try:
        response = ollama.chat(
            model='deepseek-r1:1.5b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={
                'temperature': 0.05,
                'num_ctx': 4096,
                'max_tokens': 6000,
                'stop': ["\n```"]  # Stops generation at the end of the JSON block
            }
        )
        
        # Extract JSON from markdown code block
        raw_response = response['message']['content']
        print(raw_response)
        return json.loads(raw_response)
    
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Parsing error: {str(e)}")
        return {"error": "Failed to process model response"}
    except Exception as e:
        print(f"API error: {str(e)}")
        return {"error": "Model inference failed"}


if __name__=="__main__":
    # Benchmark and test
    start_time = time.time()

    input_data = {'basic_info': {'person_id': 'JC-287336X', 'last_update': '2025-02-05T16:59:56.137876', 'BMI': 26.4}, 'risk_scores': {'scores': {'chd_risk': 14, 'base_ascvd_risk': 7.19, 'enhanced_ascvd_risk': 12.94, 'Total_Stroke_Points': 56, 'base_stroke_risk': 4.8, 'adjusted_stroke_risk': 7.1, 'stroke_risk_category': 'Moderate', 'Total_Diabetes_Points': 21, 'base_diabetes_risk': 'High', 'base_diabetes_risk_type': '(1 in 3)', 'advice_diabetes': 'Get fasting blood glucose test immediately', 'ckd_risk_points': 88.07748547883388, 'ckd_risk_group': 1, 'ckd_risk_category': 'G2: Mildly Decreased', 'risk_category': 'Low', 'overall_risk_percentage': 17.808}}, 'historical_data': [{'index': 0, 'Pers id': 'JC-287336X', 'Name': 'Pradeep Kumar', 'Rank': 'Sub', 'DOB': 0.0, 'Age': 46, 'Sex': 'M', 'Ht(cm)': 172, 'Wt(kg)': 76, 'BMI': 25.7, 'Waist Circumference': 96, 'Smoking': 'Occasionally', 'Alcohol intake': 'Occasionally', 'Phy activity': 'Active', 'Family H/O DM': 'Yes', 'Family H/O heart disease': 'No', 'Heart rate': 84, 'SBP': 128, 'DBP': 84, 'Spo2': 99, 'Temp(_f)': 97, 'Hb': 15.0, 'WBC': 8.2, 'Plt': 2.8, 'FBS': 92, 'PPBS': 114, 'HbA1c': 5.8, 'T.choles': 196, 'HDL': 46, 'LDL': 122, 'T.Bilirubin': 1.0, 'Sr.Urea': 34, 'Sr.Creatinine': 1.0, 'T.Protein': 6.2, 'Sr.Albumin': 3.8, 'Sr.Globulin': 1.8, 'chd_risk': 11, 'base_ascvd_risk': 6.8, 'enhanced_ascvd_risk': 12.24, 'Total_Stroke_Points': 55, 'base_stroke_risk': 4.3, 'adjusted_stroke_risk': 6.5, 'stroke_risk_category': 'Moderate', 'Total_Diabetes_Points': 21, 'base_diabetes_risk': 'High', 'base_diabetes_risk_type': '(1 in 3)', 'advice_diabetes': 'Get fasting blood glucose test immediately', 'ckd_risk_points': 67.72538043152912, 'ckd_risk_group': 1, 'ckd_risk_category': 'G2: Mildly Decreased', 'overall_risk_percentage': 16.948, 'risk_category': 'Low'}, {'index': 1, 'Pers id': 'JC-287336X', 'Name': 'JC-287336X', 'Rank': 'Sub', 'DOB': 0.0, 'Age': 47, 'Sex': 'M', 'Ht(cm)': 172, 'Wt(kg)': 78, 'BMI': 26.4, 'Waist Circumference': 96, 'Smoking': 'Occasionally', 'Alcohol intake': 'Occasionally', 'Phy activity': 'Active', 'Family H/O DM': 'Yes', 'Family H/O heart disease': 'No', 'Heart rate': 86, 'SBP': 130, 'DBP': 86, 'Spo2': 99, 'Temp(_f)': 97, 'Hb': 16.0, 'WBC': 7.7, 'Plt': 2.2, 'FBS': 90, 'PPBS': 112, 'HbA1c': 6.0, 'T.choles': 192, 'HDL': 44, 'LDL': 114, 'T.Bilirubin': 0.8, 'Sr.Urea': 32, 'Sr.Creatinine': 0.8, 'T.Protein': 5.6, 'Sr.Albumin': 5.0, 'Sr.Globulin': 2.0, 'chd_risk': 14, 'base_ascvd_risk': 7.19, 'enhanced_ascvd_risk': 12.94, 'Total_Stroke_Points': 56, 'base_stroke_risk': 4.8, 'adjusted_stroke_risk': 7.1, 'stroke_risk_category': 'Moderate', 'Total_Diabetes_Points': 21, 'base_diabetes_risk': 'High', 'base_diabetes_risk_type': '(1 in 3)', 'advice_diabetes': 'Get fasting blood glucose test immediately', 'ckd_risk_points': 88.07748547883388, 'ckd_risk_group': 1, 'ckd_risk_category': 'G2: Mildly Decreased', 'overall_risk_percentage': 17.808, 'risk_category': 'Low'}]}

    results = generate_health_insights(input_data)

    print(f"\nTime taken: {time.time() - start_time:.2f}s")
    # print(json.dumps(results, indent=2, ensure_ascii=False))