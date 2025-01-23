import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import plotly.graph_objects as go

class HealthMetrics:
    def __init__(self, person_id):
        self.person_id = person_id
        self.current_date = datetime.now().date()
        self.health_data = self.load_health_data()
        
    def load_health_data(self):
        try:
            df = pd.read_csv("combined_health_data_check.csv")
        except FileNotFoundError:
            raise FileNotFoundError("CSV file not found. Please check the file path.")

        person_data = df[df['person_id'] == self.person_id].copy()
        
        if person_data.empty:
            raise ValueError(f"No data found for person ID: {self.person_id}")

        person_data['date'] = pd.to_datetime(person_data['date'])
        person_data.sort_values('date', inplace=True)
        person_data.set_index('date', inplace=True)
        person_data = person_data.ffill().infer_objects(copy=False)
        
        return person_data.to_dict(orient='index')

    def calculate_age(self, date_of_birth):
        birth_date = pd.to_datetime(date_of_birth).date()
        return (self.current_date - birth_date).days // 365

    def calculate_bmi(self):
        latest = next(iter(self.health_data.values()))
        return round(latest['weight'] / (latest['height'] ** 2), 1)

    def heart_disease_risk_score(self, entry):
        score = 0
        
        # Blood Pressure
        if entry['bp_upper_limit'] >= 160 or entry['bp_lower_limit'] >= 100:
            score += 3  # Stage 2 Hypertension
        elif entry['bp_upper_limit'] >= 140 or entry['bp_lower_limit'] >= 90:
            score += 2  # Stage 1 Hypertension
        elif entry['bp_upper_limit'] >= 130 or entry['bp_lower_limit'] >= 85:
            score += 1  # Elevated BP

        # Cholesterol
        if entry['cholesterol'] > 240:
            score += 2  # High total cholesterol
        elif entry['cholesterol'] > 200:
            score += 1  # Borderline high

        if entry['cholesterol_HDL'] < 40:
            score += 2  # Low HDL (bad)
        elif entry['cholesterol_HDL'] < 60:
            score += 1  # Borderline HDL

        if entry['cholesterol_LDL'] > 190:
            score += 3  # Very high LDL
        elif entry['cholesterol_LDL'] > 160:
            score += 2  # High LDL
        elif entry['cholesterol_LDL'] > 130:
            score += 1  # Borderline high LDL

        # Smoking
        if entry['smoking'] == 'Daily':
            score += 3
        elif entry['smoking'] == 'Weekly':
            score += 2
        elif entry['smoking'] == 'Occasionally':
            score += 1

        # Physical Activity
        if entry['physical_activity_level'] == 'Never':
            score += 2
        elif entry['physical_activity_level'] == 'Occasionally':
            score += 1

        # Age
        age = self.calculate_age(entry['date_of_birth'])
        if age > 65:
            score += 3
        elif age > 50:
            score += 2
        elif age > 40:
            score += 1

        # BMI
        bmi = self.calculate_bmi()
        if bmi > 35:
            score += 3  # Obese
        elif bmi > 30:
            score += 2  # Overweight
        elif bmi > 25:
            score += 1  # Slightly overweight

        # Normalize score to 0-1
        return score / 21

    def diabetic_risk_score(self, entry):
        score = 0
        
        # Glucose Levels
        if entry['glucose_level'] > 200:
            score += 3  # Diabetic range
        elif entry['glucose_level'] > 140:
            score += 2  # Prediabetic range
        elif entry['glucose_level'] > 100:
            score += 1  # Borderline high

        # Insulin Levels
        if entry['insulin'] > 25:
            score += 2  # High insulin resistance
        elif entry['insulin'] > 15:
            score += 1  # Moderate insulin resistance

        # BMI
        bmi = self.calculate_bmi()
        if bmi > 35:
            score += 3  # Obese
        elif bmi > 30:
            score += 2  # Overweight
        elif bmi > 25:
            score += 1  # Slightly overweight

        # Physical Activity
        if entry['physical_activity_level'] == 'Never':
            score += 2
        elif entry['physical_activity_level'] == 'Occasionally':
            score += 1

        # Age
        age = self.calculate_age(entry['date_of_birth'])
        if age > 65:
            score += 3
        elif age > 50:
            score += 2
        elif age > 40:
            score += 1

        # Normalize score to 0-1
        return score / 13

    def check_concerns(self):
        df = pd.DataFrame.from_dict(self.health_data, orient='index')
        concerns = {}
        concerns['BMI'] = 'high' if (bmi := self.calculate_bmi()) > 25 else 'low' if bmi < 18.5 else 'normal'
        concerns['Heart Rate'] = 'high' if (hr := df['heart_rate'].mean()) > 100 else 'low' if hr < 60 else 'normal'
        concerns['BP'] = 'high' if (df['bp_upper_limit'].mean() > 140) or (df['bp_lower_limit'].mean() > 90) else 'normal'
        concerns['SPO2'] = 'low' if (spo := df['spo2'].mean()) < 95 else 'normal'
        concerns['Body Temp'] = 'high' if (temp := df['body_temperature'].mean()) > 37.5 else 'low' if temp < 36 else 'normal'
        concerns['Cholesterol'] = 'high' if (chol := df['cholesterol'].mean()) > 200 else 'normal'
        concerns['Hydration'] = 'low' if (hyd := df['hydration'].mean()) < 2.0 else 'normal'
        concerns['Sleep'] = 'low' if (sleep := df['sleep_duration'].mean()) < 6 else 'normal'
        concerns['Steps'] = 'low' if (steps := df['steps_taken'].mean()) < 5000 else 'normal'
        return concerns

    def generate_health_tips(self):
        tips = []
        concerns = self.check_concerns()
        
        tip_map = {
            'BMI': {
                'high': ["Increase physical activity", "Reduce processed food intake", "Consult a nutritionist"],
                'low': ["Increase calorie intake", "Focus on nutrient-dense foods", "Eat smaller, frequent meals"]
            },
            'Heart Rate': {
                'high': ["Practice deep breathing", "Avoid stimulants", "Consult a cardiologist"],
                'low': ["Stay hydrated", "Check electrolyte levels", "Monitor heart rate regularly"]
            },
            'BP': {
                'high': ["Reduce sodium intake", "Monitor blood pressure regularly", "Limit alcohol consumption"],
                'low': ["Increase fluid intake", "Check for underlying conditions", "Avoid sudden posture changes"]
            },
            'SPO2': {
                'low': ["Improve ventilation", "Practice breathing exercises", "Avoid smoking"]
            },
            'Hydration': {
                'low': ["Drink at least 2 liters of water daily", "Include hydrating foods like fruits", "Avoid excessive caffeine"]
            },
            'Sleep': {
                'low': ["Maintain a consistent sleep schedule", "Avoid screens before bed", "Create a relaxing bedtime routine"]
            },
            'Steps': {
                'low': ["Take short walks every hour", "Use stairs instead of elevators", "Set daily step goals"]
            }
        }
        
        for metric, status in concerns.items():
            if status in ['high', 'low'] and metric in tip_map:
                tips.extend([f"{metric}: {tip}" for tip in tip_map[metric].get(status, [])])
        
        return tips if tips else ["No specific recommendations - maintain current healthy habits"]

class EnhancedHealthMetrics(HealthMetrics):
    def __init__(self, person_id):
        super().__init__(person_id)
        self.metrics_df = pd.DataFrame.from_dict(self.health_data, orient='index')
        self.perform_clustering()
        
    def perform_clustering(self):
        # Calculate risk scores
        self.metrics_df['heart_risk'] = self.metrics_df.apply(
            lambda x: self.heart_disease_risk_score(x), axis=1
        )
        self.metrics_df['diabetic_risk'] = self.metrics_df.apply(
            lambda x: self.diabetic_risk_score(x), axis=1
        )
        
        # Select features for clustering
        features = ['heart_rate', 'bp_upper_limit', 'spo2', 'body_temperature', 'cholesterol',
                    'haemoglobin', 'thyroid_levels', 'glucose_level', 'insulin', 'cough',
                    'fever', 'sore_throat', 'shortness_of_breath', 'headache', 'hydration',
                    'sleep_duration', 'steps_taken']
        
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.metrics_df[features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.metrics_df['cluster'] = kmeans.fit_predict(scaled_data)
        
        # Calculate clustering risk score (normalized to 0-1)
        self.metrics_df['cluster_risk'] = self.metrics_df['cluster'].apply(
            lambda x: x / 2  # Normalize cluster number to 0-1
        )
        
        # Calculate overall risk score
        self.metrics_df['overall_risk'] = (
            self.metrics_df['heart_risk'] + 
            self.metrics_df['diabetic_risk'] + 
            self.metrics_df['cluster_risk']
        ) / 3
        
        # Label risk levels based on thresholds
        self.metrics_df['risk_level'] = self.metrics_df['overall_risk'].apply(
            lambda x: 'High Risk' if x > 0.55 else 'Moderate Risk' if x >= 0.20 else 'Low Risk'
        )
        
    def get_last_5_days_status(self):
        last_5_days = self.metrics_df.tail(5).copy()
        last_5_days['date'] = last_5_days.index.strftime('%Y-%m-%d')
        last_5_days = last_5_days[['date', 'cluster_risk', 'heart_risk', 'diabetic_risk', 'overall_risk', 'risk_level']]
        return last_5_days.to_dict(orient='records')
    
    def plot_risk_trends(self):
        last_5_days = self.metrics_df.tail(5).copy()
        last_5_days['date'] = last_5_days.index.strftime('%Y-%m-%d')
        
        # Create figure
        fig = go.Figure()

        # Add traces for each risk type
        fig.add_trace(go.Scatter(
            x=last_5_days['date'], y=last_5_days['cluster_risk'],
            mode='lines+markers', name='Cluster Risk'
        ))
        fig.add_trace(go.Scatter(
            x=last_5_days['date'], y=last_5_days['heart_risk'],
            mode='lines+markers', name='Heart Risk'
        ))
        fig.add_trace(go.Scatter(
            x=last_5_days['date'], y=last_5_days['diabetic_risk'],
            mode='lines+markers', name='Diabetic Risk'
        ))
        fig.add_trace(go.Scatter(
            x=last_5_days['date'], y=last_5_days['overall_risk'],
            mode='lines+markers', name='Overall Risk'
        ))

        # Add risk level background colors
        fig.add_hrect(y0=0, y1=0.20, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Low Risk")
        fig.add_hrect(y0=0.20, y1=0.55, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Moderate Risk")
        fig.add_hrect(y0=0.55, y1=1.0, line_width=0, fillcolor="red", opacity=0.1, annotation_text="High Risk")

        # Update layout
        fig.update_layout(
            title="Last 5 Instances Risk Trends",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            yaxis_range=[0, 1],
            legend=dict(x=0.02, y=0.98),
            hovermode="x unified",
            width=1400,  # Set the width of the figure
            height=1000
        )
        # fig.show()
        fig.write_image('figure.png', engine='kaleido', scale=2)
    
    def get_health_status(self):
        status = {
            'current_risk': self.metrics_df['risk_level'].iloc[-1],
            'trend': self.calculate_trend(),
            'anomalies': self.detect_anomalies(),
            'scores': {
                'heart_risk': self.metrics_df['heart_risk'].iloc[-1],
                'diabetic_risk': self.metrics_df['diabetic_risk'].iloc[-1],
                'cluster_risk': self.metrics_df['cluster_risk'].iloc[-1],
                'overall_risk': self.metrics_df['overall_risk'].iloc[-1]
            }
        }
        return status
    
    def calculate_trend(self):
        # Calculate 7-day moving average of overall risk
        return self.metrics_df['overall_risk'].rolling(7).mean().iloc[-1]
    
    def detect_anomalies(self):
        # Use isolation forest for anomaly detection
        iso = IsolationForest(contamination=0.1)
        self.metrics_df['anomaly'] = iso.fit_predict(self.metrics_df[['heart_rate', 'glucose_level']])
        return self.metrics_df[self.metrics_df['anomaly'] == -1].index.strftime('%Y-%m-%d').tolist()
    
    def cluster_analysis(self):
        analysis = {
            'risk_distribution': self.metrics_df['risk_level'].value_counts().to_dict(),
            'avg_heart_risk': self.metrics_df.groupby('risk_level')['heart_risk'].mean().to_dict(),
            'avg_diabetic_risk': self.metrics_df.groupby('risk_level')['diabetic_risk'].mean().to_dict(),
            'avg_cluster_risk': self.metrics_df.groupby('risk_level')['cluster_risk'].mean().to_dict(),
            'avg_overall_risk': self.metrics_df.groupby('risk_level')['overall_risk'].mean().to_dict()
        }
        return analysis
    
    def generate_report(self):
        report = {
            'basic_info': {
                'person_id': self.person_id,
                'age': self.calculate_age(next(iter(self.health_data.values()))['date_of_birth']),
                'bmi': self.calculate_bmi()
            },
            'health_status': self.get_health_status(),
            'last_5_instances_status': self.get_last_5_days_status(),
            'cluster_analysis': self.cluster_analysis(),
            'recommendations': self.generate_health_tips()
        }
        return report

# Driver Code
if __name__ == "__main__":
    try:
        person_id = int(input("Enter person ID (1-6): "))
        analyzer = EnhancedHealthMetrics(person_id)
        report = analyzer.generate_report()
        print(report)
        print("\n=== Health Report ===")
        print(f"Person ID: {report['basic_info']['person_id']}")
        print(f"Age: {report['basic_info']['age']}")
        print(f"BMI: {report['basic_info']['bmi']}")
        
        print("\nCurrent Risk Level:", report['health_status']['current_risk'])
        print("Recent Risk Trend:", f"{report['health_status']['trend']*100:.2f} % (7-day avg)")
        print("Anomaly Dates:", report['health_status']['anomalies'] or "None detected")
        
        print("\nRisk Scores:")
        print(f"- Heart Risk: {report['health_status']['scores']['heart_risk']*100:.2f} % ")
        print(f"- Diabetic Risk: {report['health_status']['scores']['diabetic_risk']*100:.2f} % ")
        print(f"- Cluster Risk: {report['health_status']['scores']['cluster_risk']*100:.2f} % ")
        print(f"- Overall Risk: {report['health_status']['scores']['overall_risk']*100:.2f} % ")
        
        print("\nLast 5 instances Status:")
        for day in report['last_5_instances_status']:
            print(f"{day['date']}: Cluster Risk={day['cluster_risk']*100:.2f} % , Heart Risk={day['heart_risk']*100:.2f} % , "
                  f"Diabetic Risk={day['diabetic_risk']*100:.2f} % , Overall Risk={day['overall_risk']*100:.2f} % , "
                  f"Risk Level={day['risk_level']}")
        
        print("\nCluster Analysis:")
        print(f"Risk Distribution: {report['cluster_analysis']['risk_distribution']}")
        # print(f"Average Heart Risk by Risk Level: {report['cluster_analysis']['avg_heart_risk']}")
        # print(f"Average Diabetic Risk by Risk Level: {report['cluster_analysis']['avg_diabetic_risk']}")
        # print(f"Average Cluster Risk by Risk Level: {report['cluster_analysis']['avg_cluster_risk']}")
        # print(f"Average Overall Risk by Risk Level: {report['cluster_analysis']['avg_overall_risk']}")
        
        print("\nRecommendations:")
        for tip in report['recommendations']:
            print(f"- {tip}")
        
        # Plot risk trends
        analyzer.plot_risk_trends()
        
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")