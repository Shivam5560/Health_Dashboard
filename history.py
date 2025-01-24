import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

class HealthHistoryAnalyzer:
    def __init__(self, person_id):
        self.person_id = person_id
        self.df = self._load_and_preprocess()
        self.person_data = self._filter_person_data()
        
    def _load_and_preprocess(self):
        # Load the CSV file
        df = pd.read_csv("combined_health_data_check.csv")
        
        # Convert and clean data
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean numeric columns with units (e.g., '216 mg/dL' -> 216.0)
        numeric_cols = ['height', 'weight', 'heart_rate', 'cholesterol', 
                       'glucose_level', 'body_temperature', 'steps_taken']
        
        for col in numeric_cols:
            if df[col].dtype == object:  # Check if the column contains strings
                df[col] = df[col].str.extract('(\d+\.?\d*)').astype(float)  # Extract numeric values
        
        # Calculate additional metrics
        df['bmi'] = df['weight'] / (df['height'] ** 2)
        df['bp_category'] = df.apply(self._categorize_bp, axis=1)
        
        return df

    def _filter_person_data(self):
        person_df = self.df[self.df['person_id'] == self.person_id]
        return person_df.sort_values('date').ffill()

    def _categorize_bp(self, row):
        systolic = row['bp_upper_limit']
        diastolic = row['bp_lower_limit']
        
        if systolic >= 140 or diastolic >= 90:
            return 'High'
        elif systolic >= 130 or diastolic >= 85:
            return 'Elevated'
        return 'Normal'

    def get_key_metrics(self):
        latest = self.person_data.iloc[-1]
        avg_values = {
            'heart_rate': self.person_data['heart_rate'].mean(),
            'systolic_bp': self.person_data['bp_upper_limit'].mean(),
            'diastolic_bp': self.person_data['bp_lower_limit'].mean(),
            'glucose': self.person_data['glucose_level'].mean(),
            'cholesterol': self.person_data['cholesterol'].mean(),
            'activity_level': self.person_data['physical_activity_level'].mode()[0]
        }
        
        return {
            'current_bmi': f"{latest['bmi']:.1f}",
            'avg_heart_rate': f"{avg_values['heart_rate']:.0f}",
            'blood_pressure': f"{latest['bp_upper_limit']:.0f}/{latest['bp_lower_limit']:.0f}",
            'bp_category': latest['bp_category'],
            'avg_glucose': f"{avg_values['glucose']:.0f}",
            'total_cholesterol': f"{avg_values['cholesterol']:.0f}",
            'activity_level': avg_values['activity_level']
        }

    

    def generate_plots(self):
        # Create subplots with 3 rows (one for each metric)
        fig = make_subplots(
            rows=3, cols=1,  # 3 rows, 1 column
            subplot_titles=("Heart Rate Trend", "Blood Pressure Trend", "Cholesterol Trend"),
            vertical_spacing=0.1,  # Space between subplots
            shared_xaxes=True  # Share the x-axis (date) across subplots
        )

        # Add Heart Rate Trend (Row 1)
        fig.add_trace(
            go.Scatter(
                x=self.person_data['date'],
                y=self.person_data['heart_rate'],
                name='Heart Rate',
                line=dict(color='#4CAF50')
            ),
            row=1, col=1  # Place in the first row
        )

        # Add Systolic BP Trend (Row 2)
        fig.add_trace(
            go.Scatter(
                x=self.person_data['date'],
                y=self.person_data['bp_upper_limit'],
                name='Systolic BP',
                line=dict(color='#FF5252')
            ),
            row=2, col=1  # Place in the second row
        )

        # Add Diastolic BP Trend (Row 2)
        fig.add_trace(
            go.Scatter(
                x=self.person_data['date'],
                y=self.person_data['bp_lower_limit'],
                name='Diastolic BP',
                line=dict(color='#FF4081')
            ),
            row=2, col=1  # Place in the second row (same as Systolic BP)
        )

        # Add Cholesterol Trend (Row 3)
        fig.add_trace(
            go.Scatter(
                x=self.person_data['date'],
                y=self.person_data['cholesterol'],
                name='Cholesterol',
                line=dict(color='#FFA726')
            ),
            row=3, col=1  # Place in the third row
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Cardiovascular Health Trends Over Time',
            height=800,  # Adjust height for better visibility
            template='plotly_dark',
            showlegend=True,
            hovermode='x unified'  # Show hover info for all traces
        )

        # Customize y-axis titles
        fig.update_yaxes(title_text="Heart Rate (bpm)", row=1, col=1)
        fig.update_yaxes(title_text="Blood Pressure (mmHg)", row=2, col=1)
        fig.update_yaxes(title_text="Cholesterol (mg/dL)", row=3, col=1)

        # Customize x-axis title (only for the bottom subplot)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    def generate_insights(self):
        metrics = self.get_key_metrics()
        insights = []
        
        # BMI Analysis
        bmi = float(metrics['current_bmi'])
        if bmi < 18.5:
            insights.append("Underweight BMI detected - recommend nutritional consultation")
        elif bmi > 25:
            insights.append("Elevated BMI observed - suggest dietary and exercise planning")

        # Blood Pressure Insights
        if metrics['bp_category'] == 'High':
            insights.append("Consistently high blood pressure readings - urgent clinical review needed")
            
        # Activity Level Correlation
        if metrics['activity_level'] == 'Never' and bmi > 25:
            insights.append("Sedentary lifestyle contributing to weight management issues")

        # Cholesterol Check
        if float(metrics['total_cholesterol']) > 200:
            insights.append("Elevated cholesterol levels - recommend lipid profile testing")
            
        return {
            'metrics': metrics,
            'insights': insights,
            'plots': self.generate_plots()
        }