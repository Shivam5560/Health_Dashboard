import pandas as pd
import numpy as np
import math
from flask import Flask, request, jsonify
import json
from generate import generate_health_insights
import time


app = Flask(__name__)

class HealthMetrics:
    def __init__(self,df):
        self.health_data = df
    

    def calculate_kidney_risk(self, row):
        creatinine = row['sr_creatinine']
        age = row['age']
        sex = row['sex']
        
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
        gender = entry['sex']
        age = entry['age']
        ldl_tc_level = entry['ldl']
        hdl_level = entry['hdl']
        systolic_bp = entry['sbp']
        diastolic_bp = entry['dbp']
        has_diabetes = entry['family_h_o_dm']
        is_smoker = entry['smoking']

        total_points = 0
    

        # Add points for age
        for (min_age, max_age), points in point_systems[gender]['age'].items():
            if min_age <= age <= max_age:
                total_points += points
                break

        # Add points for ldl/TC level
        for (min_level, max_level), points in point_systems[gender]['ldl_level'].items():
            if min_level <= ldl_tc_level <= max_level:
                total_points += points
                break

        # Add points for hdl level
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
            'ln_age': 2.891 if entry['sex'] == 'M' else 2.635,
            'ln_tc': 1.115,
            'ln_hdl': -0.162,
            'ln_sbp': 0.498,
            'smoker': 0.702,
            'diabetes': 0.314,
            'waist_cm': 0.017,
            'hba1c': 0.021 if entry['hba1c'] >= 5.7 else 0
        }

        # Variable transformations
        ln_age = np.log(entry['age'])
        ln_tc = np.log(entry['t_choles'])
        ln_hdl = np.log(entry['hdl'])
        ln_sbp = np.log(entry['sbp'])
        smoker = 1 if entry['smoking'] != 'Never' else 0
        family_dm = 1 if entry['family_h_o_dm'] == 'Yes' else 0  # Corrected variable name

        # Calculate risk sum
        risk_sum = (
            coef['ln_age'] * ln_age +
            coef['ln_tc'] * ln_tc +
            coef['ln_hdl'] * ln_hdl +
            coef['ln_sbp'] * ln_sbp +
            coef['smoker'] * smoker +
            coef['diabetes'] * family_dm +
            coef['waist_cm'] * entry['waist_circumference'] +
            coef['hba1c'] * entry['hba1c']
        )

        # Baseline survival values
        baseline_survival = 0.9012 if entry['sex'] == 'M' else 0.9215

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
        points += max(entry['age'] - 20, 0)  # 1 point/year over 20[3]
        points += 3 if entry['sex'] == 'M' else 0  # [3]
        
        # 2. Medical History
        points += 8 if entry['family_h_o_dm']=='Yes' else 0  # Diabetes[3]
        points += 5 if entry['family_h_o_heart_disease']=="Yes" else 0  # [3]
        points += 2 if entry['sbp'] >= 140 or entry['dbp'] >= 90 else 0  # Hypertension[5]
        
        # 3. Lifestyle Factors
        points += 8 if entry['smoking']!="Never" else 0  # [3]
        points += 3 if entry['alcohol_intake']!="Never" else 0  # >7 drinks/week[3]
        points += 2 if entry['phy_activity']=="Light" else 0  
        
        # 4. Metabolic Markers
        points += 7 if entry['hba1c'] >= 6.5 else (3 if 5.7 <= entry['hba1c'] < 6.5 else 0)  # [3][4]
        points += 2 if entry['bmi'] >= 25 else 0  # [5]
        points += 2 if entry['waist_circumference'] > 90 else 0  # Male threshold[4]
        
        # 5. Lipid Profile
        points += 1 if entry['ldl'] > 160 else 0  # [5]
        
        # Risk calculation formula[3]
        exponent = points / 10
        risk_percent = (1 - (S_ref ** math.exp(exponent))) * 100
        
        # Indian population multiplier (1.5x based on Asian studies[6])
        adjusted_risk = min(risk_percent * 1.5, 100)  # Cap at 100%
        
        return (points, round(risk_percent, 1), round(adjusted_risk, 1), self.categorize_stroke_risk(adjusted_risk))

    def calculate_diabetes_risk(self, entry):
        points = 0

        # 1. age Group
        age = entry['age']
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
        points += 3 if entry['sex'] == 'M' else 0

        # 3. Ethnicity
        points += 2  # Indian/Asian

        # 4. Family History
        points += 3 if entry['family_h_o_dm'] == 'Yes' else 0

        # 5. Blood Sugar Levels (fbs/ppbs)
        fbs = entry['fbs']
        ppbs = entry['ppbs']
        if fbs >= 100 or ppbs >= 140:
            points += 6

        # 6. smoking
        points += 2 if entry['smoking'] != 'Never' else 0

        # 7. Physical Activity
        points += 2 if entry['phy_activity'] == 'Light' else 0

        # 8. Waist Measurement
        is_high_risk_ethnicity = True
        waist = entry['waist_circumference']
        gender = entry['sex']
        
        if is_high_risk_ethnicity:
            if gender == 'M':
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
            if gender == 'M':
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

        # age warning for <25 years
        if age < 25 and points >= 12:
            advice += '\n*Risk may be overestimated in under-25s'

        return (points, risk, risk_type, advice)

    

class EnhancedHealthMetrics(HealthMetrics):
    def __init__(self, person_id):
        super().__init__(person_id)
        
        self.metrics_df = self.health_data
        # Assuming self.metrics_df is your DataFrame with columns 'wt_kg' and 'ht_cm'
        self.metrics_df['bmi'] = round(self.metrics_df['wt_kg'] / ((self.metrics_df['ht_cm'] / 100) ** 2), 1)
        
        self.perform_scoring()
        self.bmi  = self.metrics_df['bmi'].iloc[-1],
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
        

        #self.metrics_df.to_csv("winner.csv", index=True)
        
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
            'risk_scores': self.get_health_status()
        }
    
    def get_full_data(self):
        hist_df = self.metrics_df
        return hist_df.to_dict(orient='list')




@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Get the person_id from the request

        data = request.json
        # data_dict = json.loads(data)  # Convert JSON string to dictionary
        df = pd.DataFrame(data)  # Convert dictionary to DataFrame

        if not data:
            return jsonify({"error": "Data is required"}), 400

        # Create an instance of EnhancedHealthMetrics
        analyzer = EnhancedHealthMetrics(df)
        
        # Generate the report using the instance method
        report = analyzer.generate_report()
        
        report = json.loads(json.dumps(report, default=lambda x: int(x) if isinstance(x, np.integer) else x))
        final_json = analyzer.get_full_data()
        # Benchmark and test
        start_time = time.time()
        results = generate_health_insights(final_json)
        execution_time = time.time() - start_time

        # Convert results if needed
        results = json.loads(json.dumps(results, default=lambda x: int(x) if isinstance(x, np.integer) else x))

        # Append results directly into report
        report.update(results)
        
        

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
    


