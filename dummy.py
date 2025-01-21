import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import plotly.express as px

class HealthMetrics:
    def __init__(self, person_id):
        self.person_id = person_id
        self.health_data = self.load_health_data()

    def load_health_data(self):
        # Load data using pandas
        df = pd.read_csv('combined_health_data.csv')
        
        # Filter data for the specific person_id
        person_data = df[df['person_id'] == self.person_id]
        
        # Convert 'date' column to datetime format for proper sorting
        person_data['date'] = pd.to_datetime(person_data['date'])
        
        # Sort the DataFrame by date
        person_data = person_data.sort_values(by='date')
        
        # Handle missing values by forward filling
        person_data.fillna(method='ffill', inplace=True)
        
        # Convert the DataFrame to a dictionary with dates as keys
        data = person_data.set_index('date').T.to_dict()
        return data

    def calculate_bmi(self):
        latest_metrics = self.health_data[list(self.health_data.keys())[-1]]
        height = latest_metrics['height']
        weight = latest_metrics['weight']
        return round(weight / (height ** 2), 1)

    def check_concerns(self):
        concerns = {}
        
        # Create a DataFrame for analysis
        metrics_df = pd.DataFrame.from_dict(self.health_data, orient='index')
        
        # Calculate current metrics
        bmi = self.calculate_bmi()
        concerns['BMI'] = 'high' if bmi > 25 else 'low' if bmi < 18.5 else 'normal'
        
        avg_heart_rate = metrics_df['heart_rate'].mean()
        concerns['Heart Rate'] = 'high' if avg_heart_rate > 100 else 'low' if avg_heart_rate < 60 else 'normal'
        
        avg_bp_upper = metrics_df['bp_upper_limit'].mean()
        avg_bp_lower = metrics_df['bp_lower_limit'].mean()
        concerns['BP'] = 'high' if avg_bp_upper > 140 or avg_bp_lower > 90 else 'normal'
        
        avg_spo2 = metrics_df['spo2'].mean()
        concerns['SPO2'] = 'low' if avg_spo2 < 95 else 'normal'
        
        avg_body_temp = metrics_df['body_temperature'].mean()
        concerns['Body Temperature'] = 'high' if avg_body_temp > 37.5 else 'low' if avg_body_temp < 36 else 'normal'
        
        avg_cholesterol = metrics_df['cholesterol'].mean()
        concerns['Cholesterol'] = 'high' if avg_cholesterol > 200 else 'normal'
        
        return concerns

    def generate_health_tips(self):
        tips = []
        concerns = self.check_concerns()
        
        # Detailed do's and don'ts based on health metrics
        if concerns['BMI'] == 'high':
            tips.append("Do: Engage in regular physical activity and maintain a balanced diet rich in fruits and vegetables.")
            tips.append("Don't: Skip meals or rely on fad diets.")
        
        if concerns['Heart Rate'] == 'high':
            tips.append("Do: Practice relaxation techniques such as deep breathing or yoga.")
            tips.append("Don't: Ignore persistent high heart rates; consult a healthcare professional.")
        
        if concerns['BP'] == 'high':
            tips.append("Do: Reduce sodium intake and increase potassium-rich foods.")
            tips.append("Don't: Consume excessive alcohol or caffeine.")
        
        if concerns['SPO2'] == 'low':
            tips.append("Do: Ensure proper ventilation and consider breathing exercises.")
            tips.append("Don't: Smoke or expose yourself to secondhand smoke.")
        
        if concerns['Body Temperature'] == 'high':
            tips.append("Do: Stay hydrated and rest adequately.")
            tips.append("Don't: Engage in strenuous activities when feeling unwell.")
        
        if concerns['Cholesterol'] == 'high':
            tips.append("Do: Include healthy fats in your diet, like avocados and nuts.")
            tips.append("Don't: Consume trans fats found in many processed foods.")

        return tips

    def analyze_health_impacts(self):
        impacts = []
        metrics_df = pd.DataFrame.from_dict(self.health_data, orient='index')
        
        # Clustering to identify health states
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(metrics_df[['heart_rate', 'bp_upper_limit', 'spo2', 'body_temperature', 'cholesterol','haemoglobin','thyroid_levels','glucose_level','insulin','cough','fever','sore_throat','shortness_of_breath','headache']])
        kmeans = KMeans(n_clusters=3)
        metrics_df['health_state'] = kmeans.fit_predict(scaled_data)
        
        # Analyze cluster centroids
        centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                 columns=['heart_rate', 'bp_upper_limit', 'spo2', 'body_temperature', 'cholesterol','haemoglobin','thyroid_levels','glucose_level','insulin','cough','fever','sore_throat','shortness_of_breath','headache'])
        centroids['cluster'] = range(3)
        
        # Assign meaningful labels to clusters
        cluster_labels = {
            0: "High-risk state (high BP, high cholesterol, low SPO2)",
            1: "Moderate-risk state (slightly elevated metrics)",
            2: "Low-risk state (optimal health)"
        }
        
        # Add cluster descriptions to impacts
        for cluster, label in cluster_labels.items():
            impacts.append(f"Cluster {cluster}: {label}. Count: {metrics_df['health_state'].value_counts().get(cluster, 0)}")
        
        # Add centroid details for better interpretation
        impacts.append("Cluster Centroids (average values):")
        impacts.append(centroids.to_string())
        
        return impacts
        
    
    def generate_insights(self):
        insights = {
            "Cumulative Metrics": {
                "Average Weight": pd.Series([data["weight"] for data in self.health_data.values()]).mean(),
                "Average Heart Rate": pd.Series([data["heart_rate"] for data in self.health_data.values()]).mean(),
                "Average BP Upper Limit": pd.Series([data["bp_upper_limit"] for data in self.health_data.values()]).mean(),
                "Average BP Lower Limit": pd.Series([data["bp_lower_limit"] for data in self.health_data.values()]).mean(),
                "Average SPO2": pd.Series([data["spo2"] for data in self.health_data.values()]).mean(),
                "Average Body Temperature": pd.Series([data["body_temperature"] for data in self.health_data.values()]).mean(),
                "Average Cholesterol": pd.Series([data["cholesterol"] for data in self.health_data.values()]).mean(),
                "Average Glucose Level": pd.Series([data["glucose_level"] for data in self.health_data.values()]).mean(),
                "Average Urea": pd.Series([data["urea"] for data in self.health_data.values()]).mean(),
            },
            "Current Concerns": self.check_concerns(),
            "Health Tips": self.generate_health_tips(),
            "Health Impact Analysis": self.analyze_health_impacts()
        }
        
        return insights


    def get_last_14_days_status(self):
        metrics_df = pd.DataFrame.from_dict(self.health_data, orient='index')
        
        # List of all 14 features
        features = ['heart_rate', 'bp_upper_limit', 'spo2', 'body_temperature', 'cholesterol',
                    'haemoglobin', 'thyroid_levels', 'glucose_level', 'insulin', 'cough',
                    'fever', 'sore_throat', 'shortness_of_breath', 'headache']
        
        # Clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(metrics_df[features])
        kmeans = KMeans(n_clusters=3)
        metrics_df['health_state'] = kmeans.fit_predict(scaled_data)
        
        # Assign meaningful labels to clusters
        cluster_labels = {
            0: "High-risk state",
            1: "Moderate-risk state",
            2: "Low-risk state"
        }
        
        # Get the last 14 days' data
        last_14_days = metrics_df.tail(14)
        
        # Sort by date
        last_14_days = last_14_days.sort_index()
        
        # Extract date, cluster level, and cluster name
        last_14_days_status = last_14_days[['health_state']].copy()
        last_14_days_status['cluster_name'] = last_14_days_status['health_state'].map(cluster_labels)
        
        # Reset index to include date in the output
        last_14_days_status = last_14_days_status.reset_index()
        last_14_days_status.rename(columns={'index': 'date', 'health_state': 'cluster_level'}, inplace=True)
        
        return last_14_days_status

    def plot_pie_chart(self):
        # Get last 14 days' status
        last_14_days_status = self.get_last_14_days_status()
        
        # Count the occurrences of each cluster
        cluster_counts = last_14_days_status['cluster_name'].value_counts()
        
        # Create a pie chart using Plotly
        fig = px.pie(cluster_counts, 
                     values=cluster_counts.values, 
                     names=cluster_counts.index, 
                     title='Distribution of Health States in Last 14 Days',
                     labels={'names': 'Health State', 'values': 'Count'})
        
        # Add legends and customize layout
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend_title_text='Health State')
        
        # Show the plot
        fig.show()
    
    
# Example usage
if __name__ == '__main__':

    user_id = int(input("Enter user ID (1-3): "))
    health_metrics = HealthMetrics(person_id=user_id)

    # Generate insights and display them
    insights = health_metrics.generate_insights()
    print(insights)

    # Get last 14 days' status
    last_14_days_status = health_metrics.get_last_14_days_status()
    print("Last 14 Days' Status:")
    print(last_14_days_status.to_string(index=False))

    # Plot pie chart
    health_metrics.plot_pie_chart()