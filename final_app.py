import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import math
from plotly.subplots import make_subplots
from flask import Flask, request, jsonify
import json
import csv
app = Flask(__name__)

class HealthMetrics:
    def __init__(self, person_id):
        self.person_id = person_id
        self.current_date = datetime.now().date()
        self.health_data = self.load_health_data()
        
    def load_health_data(self):
        try:
            df = pd.read_csv("uploads/data.csv")
        except FileNotFoundError:
            raise FileNotFoundError("CSV file not found. Please check the file path.")

        person_data = df[df['Pers id'] == self.person_id].copy()
        person_data.sort_values('Age', inplace=True,ascending=True)
        print(person_data.head(4))
        if person_data.empty:
            raise ValueError(f"No data found for person ID: {self.person_id}")

        # person_data = person_data.ffill().infer_objects(copy=False)
        
        return person_data.to_dict(orient='index')


    def calculate_kidney_risk(self, row):
        creatinine = row['Sr.Creatinine']
        age = row['Age']
        sex = row['Sex']
        
        # Calculate eGFR (example uses simplified CKD-EPI formula)
        if sex == 1:  # Male
            if creatinine <= 0.9:
                egfr = 141 * (creatinine / 0.9) ** -0.411 * 0.993 ** age
            else:
                egfr = 141 * (creatinine / 0.9) ** -1.209 * 0.993 ** age
        else:  # Female
            if creatinine <= 0.7:
                egfr = 144 * (creatinine / 0.7) ** -0.329 * 0.993 ** age
            else:
                egfr = 144 * (creatinine / 0.7) ** -1.209 * 0.993 ** age

        # Determine GFR category :cite[2]:cite[4]
        if egfr >= 90:
            category = 'G1: Normal or High'
            risk_score = 0
        elif 60 <= egfr < 90:
            category = 'G2: Mildly Decreased'
            risk_score = 1
        elif 45 <= egfr < 60:
            category = 'G3a: Mildly to Moderately Decreased'
            risk_score = 2
        elif 30 <= egfr < 45:
            category = 'G3b: Moderately to Severely Decreased'
            risk_score = 3
        elif 15 <= egfr < 30:
            category = 'G4: Severely Decreased'
            risk_score = 4
        else:
            category = 'G5: Kidney Failure'
            risk_score = 5

        return (egfr,risk_score, category)




    def coronary_heart_disease_risk_score(self,entry):
        point_systems = {
            'M': {
                'age': {
                    (30, 34): -1, (35, 39): 0, (40, 44): 1, (45, 49): 2, (50, 54): 3,(55, 59): 4,(60, 64): 5,(65, 69): 6,(70, 74): 7
                },
                'ldl_level': {
                    (0, 100): -3, (100,159): 0, (160,190):1, (191, float('inf')): 2
                },
                'hdl_level': {
                    (0, 35): 2, (35, 44): 1, (45, 49): 0, (50,59):0,(60, float('inf')): -1
                },
                'bp': {
                    (0, 80): 0, (80, 84): 0, (85, 89): 1, (90,99):2, (100, float('inf')): 3
                },
                'diabetes': 2,
                'smoking': 2
            },
            'F': {
                'age': {
                    (30, 34): -9, (35, 39): -4, (40, 44): 0, (45, 49): 3, (50, 54): 6,(55, 59): 7,(60, 64): 8,(65, 69): 8,(70, 74): 8
                },
                'ldl_level': {
                    (0, 100): -2, (100,159): 0, (160,190):2, (191, float('inf')): 2
                },
                'hdl_level': {
                    (0, 35): 5, (35, 44): 2, (45, 49): 1, (50,59):0,(60, float('inf')): -2
                },
                'bp': {
                    (0, 80): -3, (80, 84): 0, (85, 89): 0, (90,99):2, (100, float('inf')): 3
                },
                'diabetes': 4,
                'smoking': 2
            }
        }
        gender = entry['Sex']
        age = entry['Age']
        ldl_tc_level = entry['LDL']
        hdl_level = entry['HDL']
        systolic_bp = entry['SBP']
        diastolic_bp = entry['DBP']
        has_diabetes = entry['Family H/O DM']
        is_smoker = entry['Smoking']

        total_points = 0
    

        # Add points for age
        for (min_age, max_age), points in point_systems[gender]['age'].items():
            if min_age <= age <= max_age:
                total_points += points
                break

        # Add points for LDL/TC level
        for (min_level, max_level), points in point_systems[gender]['ldl_level'].items():
            if min_level <= ldl_tc_level <= max_level:
                total_points += points
                break

        # Add points for HDL level
        for (min_level, max_level), points in point_systems[gender]['hdl_level'].items():
            if min_level <= hdl_level <= max_level:
                total_points += points
                break

        # Add points for blood pressure
        for (min_bp, max_bp), points in point_systems[gender]['bp'].items():
            if min_bp <= systolic_bp <= max_bp and min_bp <= diastolic_bp <= max_bp:
                total_points += points
                break

        # Add points for diabetes
        total_points += point_systems[gender]['diabetes'] if has_diabetes != 'Never' else 0

        # Add points for smoking
        total_points += point_systems[gender]['smoking'] if is_smoker != 'Never' else 0

        ten_year_risk = 0

        if gender=='M':
            risk_lookup_male = {
            -2: 2, -1: 2, 0: 3, 1: 4, 2: 4, 3:6, 4: 7, 5: 9, 6: 11,
            7: 14, 8: 18, 9: 22, 10: 27, 11: 33, 12: 40, 13: 47
            }
            if total_points<=-3:
                ten_year_risk = 1
            elif total_points>=14:
                ten_year_risk = 56
            else:
                ten_year_risk =  risk_lookup_male[total_points]
            

        else:
            risk_lookup_female = {
            -1: 2, 0: 2, 1: 2, 2: 3, 3:3, 4: 4, 5: 5, 6: 6,
            7: 7, 8: 8, 9: 9, 10: 11, 11: 13, 12: 15, 13: 17,14:20,15:24,16:27,
            }
            if total_points<=-2:
                ten_year_risk = 1
            elif total_points>=17:
                ten_year_risk = 32
            else:
                ten_year_risk =  risk_lookup_female[total_points]

        return ten_year_risk

    def ascvd_risk_score(self, entry):
        # Coefficients remain the same
        coef = {
            'ln_age': 2.891 if entry['Sex'] == 'M' else 2.635,
            'ln_tc': 1.115,
            'ln_hdl': -0.162,
            'ln_sbp': 0.498,
            'smoker': 0.702,
            'diabetes': 0.314,
            'waist_cm': 0.017,
            'hba1c': 0.021 if entry['HbA1c'] >= 5.7 else 0
        }

        # Variable transformations
        ln_age = np.log(entry['Age'])
        ln_tc = np.log(entry['T.choles'])
        ln_hdl = np.log(entry['HDL'])
        ln_sbp = np.log(entry['SBP'])
        smoker = 1 if entry['Smoking'] != 'Never' else 0
        family_dm = 1 if entry['Family H/O DM'] == 'Yes' else 0  # Corrected variable name

        # Calculate risk sum
        risk_sum = (
            coef['ln_age'] * ln_age +
            coef['ln_tc'] * ln_tc +
            coef['ln_hdl'] * ln_hdl +
            coef['ln_sbp'] * ln_sbp +
            coef['smoker'] * smoker +
            coef['diabetes'] * family_dm +
            coef['waist_cm'] * entry['Waist Circumference'] +
            coef['hba1c'] * entry['HbA1c']
        )

        # Baseline survival values
        baseline_survival = 0.9012 if entry['Sex'] == 'M' else 0.9215

        # Corrected: Use appropriate mean_lp (example value, adjust based on model specifics)
        mean_lp = 21.91  # Example value, replace with correct meanβX for your model
        base_risk = 1 - (baseline_survival ** np.exp(risk_sum - mean_lp))
        
        # Apply ethnicity multiplier
        enhanced_risk = base_risk * 1.8

        return round(base_risk * 100, 2), round(enhanced_risk * 100, 2)

    def categorize_stroke_risk(self,risk):
        if risk < 5:
            return 'Low'
        elif 5 <= risk < 15:
            return 'Moderate'
        else:
            return 'High'
    
    def calculate_stroke_risk(self,entry):
        # Base survival rate for reference individual
        S_ref = 0.99982  # For healthy 20yo female with no risks[3]
        
        # Risk points calculation (modified for Indian population)
        points = 0
        
        # 1. Demographic Factors
        points += max(entry['Age'] - 20, 0)  # 1 point/year over 20[3]
        points += 3 if entry['Sex'] == 'M' else 0  # [3]
        
        # 2. Medical History
        points += 8 if entry['Family H/O DM']=='Yes' else 0  # Diabetes[3]
        points += 5 if entry['Family H/O heart disease']=="Yes" else 0  # [3]
        points += 2 if entry['SBP'] >= 140 or entry['DBP'] >= 90 else 0  # Hypertension[5]
        
        # 3. Lifestyle Factors
        points += 8 if entry['Smoking']!="Never" else 0  # [3]
        points += 3 if entry['Alcohol intake']!="Never" else 0  # >7 drinks/week[3]
        points += 2 if entry['Phy activity']=="Light" else 0  
        
        # 4. Metabolic Markers
        points += 7 if entry['HbA1c'] >= 6.5 else (3 if 5.7 <= entry['HbA1c'] < 6.5 else 0)  # [3][4]
        points += 2 if entry['BMI'] >= 25 else 0  # [5]
        points += 2 if entry['Waist Circumference'] > 90 else 0  # Male threshold[4]
        
        # 5. Lipid Profile
        points += 1 if entry['LDL'] > 160 else 0  # [5]
        
        # Risk calculation formula[3]
        exponent = points / 10
        risk_percent = (1 - (S_ref ** math.exp(exponent))) * 100
        
        # Indian population multiplier (1.5x based on Asian studies[6])
        adjusted_risk = min(risk_percent * 1.5, 100)  # Cap at 100%
        
        return (points, round(risk_percent, 1), round(adjusted_risk, 1), self.categorize_stroke_risk(adjusted_risk))

    def calculate_diabetes_risk(self, entry):
        points = 0

        # 1. Age Group
        age = entry['Age']
        if age < 35:
            points += 0
        elif 35 <= age < 45:
            points += 2
        elif 45 <= age < 55:
            points += 4
        elif 55 <= age < 65:
            points += 6
        else:
            points += 8

        # 2. Gender
        points += 3 if entry['Sex'] == 'M' else 0

        # 3. Ethnicity
        points += 2  # Indian/Asian

        # 4. Family History
        points += 3 if entry['Family H/O DM'] == 'Yes' else 0

        # 5. Blood Sugar Levels (FBS/PPBS)
        fbs = entry['FBS']
        ppbs = entry['PPBS']
        if fbs >= 100 or ppbs >= 140:
            points += 6

        # 6. Smoking
        points += 2 if entry['Smoking'] != 'Never' else 0

        # 7. Physical Activity
        points += 2 if entry['Phy activity'] == 'Light' else 0

        # 8. Waist Measurement
        is_high_risk_ethnicity = True
        waist = entry['Waist Circumference']
        gender = entry['Sex']
        
        if is_high_risk_ethnicity:
            if gender == 'Male':
                if waist < 90:
                    points += 0
                elif 90 <= waist <= 100:
                    points += 4
                else:
                    points += 7
            else:  # Female
                if waist < 80:
                    points += 0
                elif 80 <= waist <= 90:
                    points += 4
                else:
                    points += 7
        else:
            if gender == 'Male':
                if waist < 102:
                    points += 0
                elif 102 <= waist <= 110:
                    points += 4
                else:
                    points += 7
            else:  # Female
                if waist < 88:
                    points += 0
                elif 88 <= waist <= 100:
                    points += 4
                else:
                    points += 7

        # Risk Categorization
        if points <= 5:
            risk = 'Low'
            risk_type = '(1 in 100)'
            advice = 'Maintain healthy lifestyle'
        elif 6 <= points <= 11:
            risk = 'Moderate'
            advice = 'Discuss with doctor, improve lifestyle'
            if 6 <= points <= 8:
                risk_type = '(1 in 50)'
            else:
                risk_type = '(1 in 30)'
        else:
            risk = 'High'
            if 12 <= points <= 15:
                risk_type = '(1 in 14)'
            elif 16 <= points <= 19:
                risk_type = '(1 in 7)'
            else:
                risk_type = '(1 in 3)'
            advice = 'Get fasting blood glucose test immediately'

        # Age warning for <25 years
        if age < 25 and points >= 12:
            advice += '\n*Risk may be overestimated in under-25s'

        return (points, risk, risk_type, advice)

    

class EnhancedHealthMetrics(HealthMetrics):
    def __init__(self, person_id):
        super().__init__(person_id)
        self.metrics_df = pd.DataFrame.from_dict(self.health_data, orient='index')
        # Assuming self.metrics_df is your DataFrame with columns 'Wt(kg)' and 'Ht(cm)'
        self.metrics_df['BMI'] = round(self.metrics_df['Wt(kg)'] / ((self.metrics_df['Ht(cm)'] / 100) ** 2), 1)

        self.perform_scoring()
        self.bmi  = self.metrics_df['BMI'].iloc[-1],
        self.chd_risk = self.metrics_df['chd_risk'].iloc[-1],
        self.base_ascvd_risk = self.metrics_df['base_ascvd_risk'].iloc[-1],
        self.enhanced_ascvd_risk = self.metrics_df['enhanced_ascvd_risk'].iloc[-1],
        self.Total_Stroke_Points = self.metrics_df['Total_Stroke_Points'].iloc[-1],
        self.base_stroke_risk = self.metrics_df['base_stroke_risk'].iloc[-1],
        self.adjusted_stroke_risk = self.metrics_df['adjusted_stroke_risk'].iloc[-1],
        self.stroke_risk_category = self.metrics_df['stroke_risk_category'].iloc[-1],
        self.Total_Diabetes_Points = self.metrics_df['Total_Diabetes_Points'].iloc[-1],
        self.base_diabetes_risk = self.metrics_df['base_diabetes_risk'].iloc[-1],
        self.base_diabetes_risk_type = self.metrics_df['base_diabetes_risk_type'].iloc[-1],
        self.advice_diabetes = self.metrics_df['advice_diabetes'].iloc[-1],
        self.ckd_risk_points = self.metrics_df['ckd_risk_points'].iloc[-1]
        self.ckd_risk_group = self.metrics_df['ckd_risk_group'].iloc[-1]
        self.ckd_risk_category = self.metrics_df['ckd_risk_category'].iloc[-1]
        self.overall_risk_percentage = self.metrics_df['overall_risk_percentage'].iloc[-1]
        self.risk_category = self.metrics_df['risk_category'].iloc[-1]

        
    def generate_recommendations(self):
        latest = self.metrics_df.iloc[-1]
        recommendations = {'do': [], 'dont': []}
        # Coronary Heart Disease Recommendations
        if latest['chd_risk'] >= 10:  # High risk threshold
            recommendations['do'].extend([
                "Consult a cardiologist for further evaluation",
                "Consider starting statin therapy if not already prescribed"
            ])
            recommendations['dont'].append("Avoid high-fat diets and smoking")
        elif 5 <= latest['chd_risk'] < 10:  # Moderate risk threshold
            recommendations['do'].append("Increase physical activity to at least 150 mins/week")
            recommendations['dont'].append("Limit saturated fats and cholesterol intake")

        # ASCVD Risk
        if latest['enhanced_ascvd_risk'] >= 7.5:  # High risk threshold
            recommendations['do'].append("Consider statin therapy after medical consultation")
            recommendations['dont'].append("Avoid prolonged sedentary behavior")

        # Stroke Prevention
        if latest['stroke_risk_category'] in ['High', 'Moderate']:
            recommendations['do'].append("Monitor atrial fibrillation status regularly")
            recommendations['dont'].append("Limit alcohol consumption to ≤2 drinks/day")

        # Chronic Kidney Disease Recommendations
        ckd_category, ckd_risk_score = latest['ckd_risk_category'],latest['ckd_risk_points']
        
        if ckd_category == 'Severe' or ckd_category == 'End Stage':
            recommendations['do'].extend([
                "Consult a nephrologist for specialized care",
                "Monitor blood pressure and blood sugar levels closely"
            ])
            recommendations['dont'].append("Avoid NSAIDs without medical supervision")
        elif ckd_category == 'Moderate':
            recommendations['do'].append("Maintain a low-protein diet as advised by your doctor")
            recommendations['dont'].append("Avoid excessive salt intake")

        # Blood pressure recommendations
        if latest['SBP'] >= 140 or latest['DBP'] >= 90:
            recommendations['do'].append("Monitor BP daily and consult cardiologist")
            recommendations['dont'].append("Avoid high-sodium foods (>1500mg/day)")
        
        # Diabetes prevention
        if latest['base_diabetes_risk'] in ['High', 'Moderate']:
            recommendations['do'].append("Get HbA1c test every 3 months")
            recommendations['do'].append("Include 30 mins aerobic exercise daily")
            recommendations['dont'].append("Avoid sugary drinks and refined carbs")
        
        # Smoking cessation
        if latest['Smoking'] != 'Never':
            recommendations['do'].append("Start smoking cessation program")
            recommendations['do'].append("Use nicotine replacement therapy")
            recommendations['dont'].append("Avoid smoking triggers like alcohol")
        
        # Weight management
        if latest['BMI'] >= 25:
            recommendations['do'].append("Aim for 5-10% weight loss gradually")
            recommendations['dont'].append("Avoid crash diets or extreme fasting")
        
        return recommendations

    def generate_visualizations(self):
        fig = make_subplots(rows=3, cols=2, subplot_titles=(
            'CHD Risk Score', 
            'Diabetes Risk Progression',
            'Stroke Risk Timeline',
            'ASCVD Risk Progression',
            'CKD Risk Progression',
        ))
        
        # Cardiovascular risk plot
        fig.add_trace(go.Line(
            x=self.metrics_df['Age'], 
            y=self.metrics_df['chd_risk'],
            name='CHD Risk Score'
        ), row=1, col=1)
        
        # Diabetes risk plot
        fig.add_trace(go.Line(
            x=self.metrics_df['Age'],
            y=self.metrics_df['Total_Diabetes_Points'],
            name='Diabetes Risk Score'
        ), row=1, col=2)
        
        # Stroke risk plot
        fig.add_trace(go.Line(
            x=self.metrics_df['Age'],
            y=self.metrics_df['adjusted_stroke_risk'],
            name='Stroke Risk'
        ), row=2, col=1)

        fig.add_trace(go.Line(
            x=self.metrics_df['Age'],
            y=self.metrics_df['enhanced_ascvd_risk'],
            name='ASCVD Risk'
        ), row=2, col=2)

        fig.add_trace(go.Line(
            x=self.metrics_df['Age'],
            y=self.metrics_df['ckd_risk_points'],
            name='CKD Risk Score'
        ), row=3, col=1)

        
        
        
        fig.update_layout(height=800, width=1200, title_text='Health Risk Dashboard')
        fig.write_image('winners.png', engine='kaleido', scale=2)

        
    def perform_scoring(self):
        # Calculate risk entry
        self.metrics_df['chd_risk'] = self.metrics_df.apply(
            lambda x: self.coronary_heart_disease_risk_score(x), axis=1
        )

        self.metrics_df[['base_ascvd_risk', 'enhanced_ascvd_risk']] = self.metrics_df.apply(
            lambda x: self.ascvd_risk_score(x), axis=1, result_type='expand'
        )
        
        self.metrics_df[['Total_Stroke_Points', 'base_stroke_risk', 'adjusted_stroke_risk', 'stroke_risk_category']] = self.metrics_df.apply(
            lambda x: self.calculate_stroke_risk(x), axis=1, result_type='expand'
        )

        self.metrics_df[['Total_Diabetes_Points', 'base_diabetes_risk', 'base_diabetes_risk_type', 'advice_diabetes']] = self.metrics_df.apply(
            lambda x: self.calculate_diabetes_risk(x), axis=1, result_type='expand'
        )

        self.metrics_df[['ckd_risk_points','ckd_risk_group' ,'ckd_risk_category']] = self.metrics_df.apply(
            lambda x: self.calculate_kidney_risk(x), axis=1,result_type='expand'
        )

        self.metrics_df[['overall_risk_percentage', 'risk_category']] = self.metrics_df.apply(
            lambda x: self.calculate_overall_risk(x), axis=1,result_type='expand'
        )
        

        self.metrics_df.to_csv("winner.csv", index=True)
        
    def calculate_overall_risk(self,entry):
        # Define weights for each risk factor
        weights = {
            'chd_risk': 0.2,
            'enhanced_ascvd_risk': 0.2,
            'adjusted_stroke_risk': 0.2,
            'base_diabetes_risk': 0.2,
            'ckd_risk_group': 0.2
        }

        # Normalize the 'base_diabetes_risk' to a percentage
        diabetes_risk_mapping = {
            'Low': 15,
            'Moderate': 25,
            'High': 35,
        }
        base_diabetes_risk = diabetes_risk_mapping.get(entry['base_diabetes_risk'], 0)

        # Normalize the 'ckd_risk_group' to a percentage
        ckd_risk_group_mapping = {
            1: 20,
            2: 40,
            3: 60,
            4: 80,
            5: 100
        }
        ckd_risk_group = ckd_risk_group_mapping.get(entry['ckd_risk_group'], 0)

        # Calculate the weighted sum of risks
        weighted_sum = (
            weights['chd_risk'] * entry['chd_risk'] +
            weights['enhanced_ascvd_risk'] * entry['enhanced_ascvd_risk'] +
            weights['adjusted_stroke_risk'] * entry['adjusted_stroke_risk'] +
            weights['base_diabetes_risk'] * base_diabetes_risk +
            weights['ckd_risk_group'] * ckd_risk_group
        )

        # Calculate the overall risk percentage
        overall_risk_percentage = weighted_sum / sum(weights.values())

        # Determine the risk category
        if overall_risk_percentage < 20:
            risk_category = 'Low'
        elif 20 <= overall_risk_percentage < 40:
            risk_category = 'Moderate'
        elif 40 <= overall_risk_percentage < 60:
            risk_category = 'High'
        else:
            risk_category = 'Very High'

        return (overall_risk_percentage, risk_category)



    def get_health_status(self):
        status = {
            'scores': {
                'chd_risk': self.metrics_df['chd_risk'].iloc[-1],
                'base_ascvd_risk': self.metrics_df['base_ascvd_risk'].iloc[-1],
                'enhanced_ascvd_risk': self.metrics_df['enhanced_ascvd_risk'].iloc[-1],
                'Total_Stroke_Points': self.metrics_df['Total_Stroke_Points'].iloc[-1],
                'base_stroke_risk': self.metrics_df['base_stroke_risk'].iloc[-1],
                'adjusted_stroke_risk': self.metrics_df['adjusted_stroke_risk'].iloc[-1],
                'stroke_risk_category': self.metrics_df['stroke_risk_category'].iloc[-1],
                'Total_Diabetes_Points':self.metrics_df['Total_Diabetes_Points'].iloc[-1],
                'base_diabetes_risk':self.metrics_df['base_diabetes_risk'].iloc[-1],
                'base_diabetes_risk_type':self.metrics_df['base_diabetes_risk_type'].iloc[-1],
                'advice_diabetes':self.metrics_df['advice_diabetes'].iloc[-1],
                'ckd_risk_points':self.metrics_df['ckd_risk_points'].iloc[-1],
                'ckd_risk_group':self.metrics_df['ckd_risk_group'].iloc[-1],
                'ckd_risk_category':self.metrics_df['ckd_risk_category'].iloc[-1],
                'risk_category':self.metrics_df['risk_category'].iloc[-1],
                'overall_risk_percentage':self.metrics_df['overall_risk_percentage'].iloc[-1],

            }
        }
        return status

    def generate_report(self):
        return {
            'basic_info': {
                'person_id': self.person_id,
                'last_update': datetime.now().isoformat(),
                'BMI': float(self.metrics_df['BMI'].iloc[-1]),
            },
            'risk_scores': self.get_health_status(),
            'historical_data': self._get_historical_data(),
            'recommendations': self.generate_recommendations(),
            # 'visualization': self.generate_visualizations()
        }
    
    def _get_historical_data(self):
        hist_df = self.metrics_df.tail(5).reset_index()
        return hist_df[[
            'Pers id','Name','Age', 'BMI', 'chd_risk', 'enhanced_ascvd_risk',
            'adjusted_stroke_risk', 'base_diabetes_risk', 'ckd_risk_points','ckd_risk_category'
        ]].to_dict(orient='records')


# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(f"{app.config['UPLOAD_FOLDER']}/{filename}")

        # Process CSV file (you can read and return its content, or do further processing)
        with open(f"{app.config['UPLOAD_FOLDER']}/{filename}", newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = list(csv_reader)

        return jsonify({"message": "File uploaded successfully", "data": rows}), 200
    
    return jsonify({"error": "Invalid file format"}), 400


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Get the person_id from the request
        data = request.json
        person_id = data.get('person_id')
        
        if not person_id:
            return jsonify({"error": "person_id is required"}), 400

        # Create an instance of EnhancedHealthMetrics
        analyzer = EnhancedHealthMetrics(person_id)
        
        # Generate the report using the instance method
        report = analyzer.generate_report()
        
        # Convert numpy.int64 to native Python int
        report = json.loads(json.dumps(report, default=lambda x: int(x) if isinstance(x, np.integer) else x))
        return jsonify(report), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

# Driver Code
if __name__ == "__main__":
    app.run(debug=True)
    


