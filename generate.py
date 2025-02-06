import ollama
from pydantic import BaseModel
from typing import List, Literal
import json
import re

class Recommendations(BaseModel):
    monitoring: List[str]
    medication: List[str]
    heart_health_recommendation: List[str]
    liver_health_recommendation: List[str]
    kidney_health_recommendation: List[str]

class Insights(BaseModel):
    primary_risk_factors: List[str]
    trend_analysis: str

class ActionItems(BaseModel):
    immediate_actions: List[str]
    long_term_goals: List[str]


class ClinicalAssessment(BaseModel):
    recommendations: Recommendations
    insights: Insights
    action_items: ActionItems
    summary: str


import re
import json




def generate_health_insights(input_data: dict) -> ClinicalAssessment:
    prompt = f"""
**Comprehensive Health Risk Assessment & Clinical Guidance**

Analyze the provided patient data and generate a **structured JSON response** with **clinically relevant recommendations** based on validated numerical insights. Ensure precise, **non-repetitive**, and **quantified** recommendations with **historical comparisons** between the latest and previous data,data is in the ascending order of date but date is not mentioned so take help of 'Age' key for better understanding of the latest and old data records/instance

#### **Patient Data:**  
{json.dumps(input_data, indent=2)}

---

### **Output Requirements**:
1. **Strict JSON Format**: Follow the exact structure of the `ClinicalAssessment` class and use different action verbs to start the sentence and quantify with data changes for all arguments/parameters/keys:
   ```json
   {{
       "recommendations": {{
           "monitoring": ["item1", "item2",..],
           "medication": ["Consider discussing medication options with a healthcare provider, including: item1",..],
            "heart_health_recommendation": ["item1 due to trends",],
            "liver_health_recommendation": ["item1 due to trends",],
            "kidney_health_recommendation": ["item1 due to trends",]
       }},
       "insights": {{
           "primary_risk_factors": ["factor1_detailed", "factor2_detailed",..],
           "trend_analysis": "Detailed analysis with quantifying changes with numbers/percentage...",
           "historical_comparisons": ["comparison1", "comparison2",...]
       }},
       "action_items": {{
           "immediate_actions": ["action1", "action2",..],
           "long_term_goals": ["goal1", "goal2",..]
       }},
       "summary": "Synopsis is 120 words precise with insights,recommendations,action items summarized",
   }}"""

    try:
        response = ollama.chat(
        model='qwen2.5:3b',
        messages=[{'role': 'user', 'content': prompt}],
        format=ClinicalAssessment.model_json_schema(),
        options={
            'temperature': 0.1,  # Reduce randomness for better structure
            'num_ctx': 4096,
            'max_tokens': 4000,
            'frequency_penalty': 1.2,  # Lowers likelihood of repeating content
        }   
    )

        # Extract JSON from markdown code block
        raw_response = response['message']['content']
        # cleaned_content = clean_json_output(raw_response)
        return json.loads(raw_response)
        
    
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Parsing error: {str(e)}")
        return {"error": "Failed to process model response"}
    except Exception as e:
        print(f"API error: {str(e)}")
        return {"error": "Model inference failed"}
