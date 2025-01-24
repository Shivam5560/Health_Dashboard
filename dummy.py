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
        if entry['cholesterol'] > 220:
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

        # Existing metrics
        concerns['BMI'] = 'high' if (bmi := self.calculate_bmi()) > 25 else 'low' if bmi < 18.5 else 'normal'
        concerns['Heart Rate'] = 'high' if (hr := df['heart_rate'].mean()) > 100 else 'low' if hr < 60 else 'normal'
        concerns['BP'] = 'high' if (df['bp_upper_limit'].mean() > 140) or (df['bp_lower_limit'].mean() > 90) else 'normal'
        concerns['SPO2'] = 'low' if (spo := df['spo2'].mean()) < 95 else 'normal'
        concerns['Body Temp'] = 'high' if (temp := df['body_temperature'].mean()) > 37.5 else 'low' if temp < 36 else 'normal'
        concerns['Cholesterol'] = 'high' if (chol := df['cholesterol'].mean()) > 200 else 'normal'
        concerns['Hydration'] = 'low' if (hyd := df['hydration'].mean()) < 2.0 else 'normal'
        concerns['Sleep'] = 'low' if (sleep := df['sleep_duration'].mean()) < 6 else 'normal'
        concerns['Steps'] = 'low' if (steps := df['steps_taken'].mean()) < 5000 else 'normal'

        # New risk metrics (assuming they are calculated elsewhere and stored in the object)
        concerns['Cluster Risk'] = self.cluster_risk  # Assuming self.cluster_risk is calculated and in range [0, 1]
        concerns['Heart Risk'] = self.heart_risk  # Assuming self.heart_risk is calculated and in range [0, 1]
        concerns['Diabetic Risk'] = self.diabetic_risk  # Assuming self.diabetic_risk is calculated and in range [0, 1]
        concerns['Overall Risk'] = self.overall_risk  # Assuming self.overall_risk is calculated and in range [0, 1]

        # Categorize risks into low, moderate, or high
        def categorize_risk(risk_value):
            if risk_value < 0.2:
                return 'low'
            elif 0.2 <= risk_value < 0.55:
                return 'moderate'
            else:
                return 'high'

        # Apply categorization to risk metrics
        for risk in ['Cluster Risk', 'Heart Risk', 'Diabetic Risk', 'Overall Risk']:
            concerns[risk] = categorize_risk(concerns[risk])

        return concerns

    def generate_health_tips(self):
        concerns = self.check_concerns()
        tips = {"do": [], "don't": []}  # Separate lists for Do's and Don'ts

        tip_map = {
            'BMI': {
                'high': {
                    'do': ["Increase physical activity", "Focus on whole, unprocessed foods", "Consult a nutritionist"],
                    'don\'t': ["Consume sugary drinks", "Eat processed or fried foods", "Skip meals"]
                },
                'low': {
                    'do': ["Increase calorie intake with healthy foods", "Eat smaller, frequent meals", "Include protein-rich foods"],
                    'don\'t': ["Skip meals", "Rely on junk food for calories", "Overeat in one sitting"]
                }
            },
            'Heart Rate': {
                'high': {
                    'do': ["Practice deep breathing exercises", "Stay hydrated", "Consult a cardiologist"],
                    'don\'t': ["Consume caffeine or stimulants", "Engage in intense exercise without medical advice", "Ignore persistent high heart rate"]
                },
                'low': {
                    'do': ["Stay hydrated", "Check electrolyte levels", "Monitor heart rate regularly"],
                    'don\'t': ["Ignore symptoms like dizziness", "Skip meals", "Engage in sudden strenuous activity"]
                }
            },
            'BP': {
                'high': {
                    'do': ["Reduce sodium intake", "Monitor blood pressure regularly", "Engage in moderate exercise"],
                    'don\'t': ["Consume alcohol excessively", "Eat processed or salty foods", "Ignore high blood pressure readings"]
                },
                'low': {
                    'do': ["Increase fluid intake", "Check for underlying conditions", "Eat small, frequent meals"],
                    'don\'t': ["Stand up too quickly", "Skip meals", "Ignore symptoms like dizziness"]
                }
            },
            'SPO2': {
                'low': {
                    'do': ["Improve ventilation in your living space", "Practice breathing exercises", "Stay hydrated"],
                    'don\'t': ["Smoke or expose yourself to secondhand smoke", "Ignore low SPO2 readings", "Stay in poorly ventilated areas"]
                }
            },
            'Hydration': {
                'low': {
                    'do': ["Drink at least 2 liters of water daily", "Include hydrating foods like fruits", "Monitor urine color for hydration"],
                    'don\'t': ["Consume excessive caffeine", "Ignore thirst signals", "Rely solely on sugary drinks"]
                }
            },
            'Sleep': {
                'low': {
                    'do': ["Maintain a consistent sleep schedule", "Create a relaxing bedtime routine", "Avoid screens before bed"],
                    'don\'t': ["Consume caffeine late in the day", "Use electronic devices in bed", "Ignore sleep deprivation symptoms"]
                }
            },
            'Steps': {
                'low': {
                    'do': ["Take short walks every hour", "Use stairs instead of elevators", "Set daily step goals"],
                    'don\'t': ["Stay sedentary for long periods", "Ignore opportunities to move", "Set unrealistic step goals"]
                }
            },
            'Cluster Risk': {
                'low': {
                    'do': ["Maintain a healthy lifestyle", "Regularly check health metrics", "Stay proactive about health"],
                    'don\'t': ["Ignore minor symptoms", "Skip regular health checkups", "Neglect mental health"]
                },
                'moderate': {
                    'do': ["Monitor symptoms closely", "Consult a healthcare provider", "Follow a structured health plan"],
                    'don\'t': ["Ignore worsening symptoms", "Delay seeking medical advice", "Neglect prescribed treatments"]
                },
                'high': {
                    'do': ["Seek immediate medical advice", "Follow a strict health regimen", "Monitor health metrics daily"],
                    'don\'t': ["Ignore severe symptoms", "Delay emergency care", "Discontinue prescribed medications"]
                }
            },
            'Heart Risk': {
                'low': {
                    'do': ["Continue healthy habits", "Monitor heart health metrics", "Stay informed about heart health"],
                    'don\'t': ["Ignore risk factors", "Skip regular checkups", "Engage in unhealthy habits"]
                },
                'moderate': {
                    'do': ["Adopt a heart-healthy diet", "Exercise regularly", "Avoid smoking and excessive alcohol"],
                    'don\'t': ["Consume high-sodium foods", "Engage in sedentary behavior", "Ignore chest pain or discomfort"]
                },
                'high': {
                    'do': ["Consult a cardiologist immediately", "Follow a strict heart-healthy plan", "Monitor heart health daily"],
                    'don\'t': ["Ignore chest pain or shortness of breath", "Delay seeking medical help", "Discontinue prescribed medications"]
                }
            },
            'Diabetic Risk': {
                'low': {
                    'do': ["Maintain a balanced diet", "Avoid excessive sugar intake", "Stay active"],
                    'don\'t': ["Consume sugary snacks or drinks", "Skip meals", "Ignore blood sugar monitoring"]
                },
                'moderate': {
                    'do': ["Monitor blood sugar levels", "Follow a diabetic-friendly diet", "Exercise regularly"],
                    'don\'t': ["Consume high-glycemic foods", "Skip prescribed medications", "Ignore symptoms like frequent urination"]
                },
                'high': {
                    'do': ["Consult a diabetologist immediately", "Follow a strict diabetic plan", "Monitor blood sugar daily"],
                    'don\'t': ["Ignore high blood sugar readings", "Delay seeking medical help", "Discontinue prescribed medications"]
                }
            },
            'Overall Risk': {
                'low': {
                    'do': ["Maintain a balanced lifestyle", "Stay proactive about health", "Regularly check health metrics"],
                    'don\'t': ["Ignore minor symptoms", "Skip regular checkups", "Neglect mental health"]
                },
                'moderate': {
                    'do': ["Consult a healthcare provider", "Follow a comprehensive health plan", "Monitor all health metrics"],
                    'don\'t': ["Ignore worsening symptoms", "Delay seeking medical advice", "Neglect prescribed treatments"]
                },
                'high': {
                    'do': ["Seek immediate medical advice", "Follow a strict health regimen", "Monitor all health metrics daily"],
                    'don\'t': ["Ignore severe symptoms", "Delay emergency care", "Discontinue prescribed medications"]
                }
            }
        }

        # Aggregate all Do's and Don'ts
        for concern, status in concerns.items():
            if concern in tip_map and status in tip_map[concern]:
                tips["do"].extend(tip_map[concern][status]['do'])
                tips["don't"].extend(tip_map[concern][status]['don\'t'])

        # Remove duplicates (if any)
        tips["do"] = list(set(tips["do"]))
        tips["don't"] = list(set(tips["don't"]))

        return tips if tips else {"message": "No specific recommendations - maintain current healthy habits"}

class EnhancedHealthMetrics(HealthMetrics):
    def __init__(self, person_id):
        super().__init__(person_id)
        self.metrics_df = pd.DataFrame.from_dict(self.health_data, orient='index')
        self.perform_clustering()
        
        # Assign risk scores as attributes
        self.cluster_risk = self.metrics_df['cluster_risk'].iloc[-1]
        self.heart_risk = self.metrics_df['heart_risk'].iloc[-1]
        self.diabetic_risk = self.metrics_df['diabetic_risk'].iloc[-1]
        self.overall_risk = self.metrics_df['overall_risk'].iloc[-1]
        
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
                    'steps_taken', 'heart_risk', 'diabetic_risk']
        
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.metrics_df[features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init to suppress warning
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
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Create 2x2 subplot grid with individual titles
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cluster Risk", "Heart Risk", 
                        "Diabetic Risk", "Overall Risk"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Helper function to create smoothed traces
        def create_trace(data, name, color):
            return go.Scatter(
                x=data['date'],
                y=data[name.lower().replace(' ', '_')],
                mode='lines+markers',
                line=dict(shape='spline', width=2.5, color=color),
                marker=dict(size=10, color=color, line=dict(width=1, color='DarkSlateGrey')),
                name=name
            )

        # Add traces to subplots
        traces = [
            (create_trace(last_5_days, 'Cluster Risk', '#1f77b4'), 1, 1),
            (create_trace(last_5_days, 'Heart Risk', '#ff7f0e'), 1, 2),
            (create_trace(last_5_days, 'Diabetic Risk', '#2ca02c'), 2, 1),
            (create_trace(last_5_days, 'Overall Risk', '#d62728'), 2, 2)
        ]

        for trace, row, col in traces:
            fig.add_trace(trace, row=row, col=col)

        # Add risk bands to all subplots
        risk_bands = [
            (0, 0.2, "green", "Low Risk"),
            (0.2, 0.55, "orange", "Moderate Risk"),
            (0.55, 1.0, "red", "High Risk")
        ]

        for row in [1, 2]:
            for col in [1, 2]:
                for y0, y1, color, text in risk_bands:
                    fig.add_hrect(
                        y0=y0, y1=y1, 
                        fillcolor=color, 
                        opacity=0.1,
                        line_width=0,
                        annotation_text=text if row == 1 and col == 1 else "",
                        annotation_position="inside top left",
                        row=row, col=col
                    )

        # Update layout settings
        fig.update_layout(
            title_text="Risk Trends Analysis - Last 5 Days",
            title_x=0.5,
            title_font_size=24,
            showlegend=False,
            margin=dict(t=100),
            width=1400,
            height=900
        )

        # Axis formatting
        fig.update_xaxes(title_text="Date", row=2, col=1, title_font_size=14)
        fig.update_xaxes(title_text="Date", row=2, col=2, title_font_size=14)
        fig.update_yaxes(
            title_text="Risk Score", 
            range=[-0.05, 1.05],  # Extended y-axis range
            title_font_size=14,
            tickfont_size=12,
            row=1, col=1
        )
        fig.update_yaxes(range=[-0.05, 1.05], row=1, col=2)  # Extended y-axis range
        fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)  # Extended y-axis range
        fig.update_yaxes(range=[-0.05, 1.05], row=2, col=2)  # Extended y-axis range

        # Save figure
        fig.write_image('risk_trends.png', engine='kaleido', scale=2)
    
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
        return self.metrics_df['overall_risk'].rolling(7).mean().iloc[-1]*100
    
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
            if tip == 'do':
                print("\nDo's\n")
            else:
                print("\nDon't\n")
            for x in report['recommendations'][tip]:
                print(f"- {x}")
        
        # Plot risk trends
        analyzer.plot_risk_trends()
        
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")