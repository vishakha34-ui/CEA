"""
Prediction Models for HealthVend System
Implements weight prediction and food category forecasting
Part of Prediction Agent in Agentic AI Architecture
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from health_calculator import HealthProfile

class PredictionModel:
    """
    Predicts future weight and nutritional needs using machine learning.
    
    Academic Context:
    This module supports the "Prediction Agent" in the Agentic AI system.
    Enables:
    - Autonomous forecasting of health outcomes
    - Proactive intervention recommendations
    - Personalized health trajectory monitoring
    - Data-driven public health decision making
    
    Models implemented:
    1. Linear Regression: Weight prediction after 7 days
    2. Category Predictor: Food category preference forecast
    """
    
    def __init__(self, user_progress_df: pd.DataFrame, users_df: pd.DataFrame):
        """
        Initialize prediction models.
        
        Args:
            user_progress_df: DataFrame with user weekly progress data
            users_df: DataFrame with user demographic data
        """
        self.user_progress_df = user_progress_df.copy()
        self.users_df = users_df.copy()
        self.weight_models = {}  # Per-user models for weight prediction
        self.scaler = StandardScaler()
        self._train_models()
    
    def _train_models(self):
        """
        Train weight prediction models for each user based on historical data.
        Uses Linear Regression on weekly weight changes and calorie intake.
        """
        # Group by user and calculate trends
        for user_id in self.user_progress_df['user_id'].unique():
            user_data = self.user_progress_df[self.user_progress_df['user_id'] == user_id].sort_values('week')
            
            if len(user_data) < 2:
                continue
            
            # Prepare features: week number and daily calories
            X = user_data[['week', 'daily_calories']].values
            
            # Calculate cumulative weight change
            y = user_data['weight_change_kg'].cumsum().values
            
            try:
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train linear regression
                model = LinearRegression()
                model.fit(X_scaled, y)
                
                self.weight_models[user_id] = {
                    'model': model,
                    'scaler': self.scaler,
                    'last_week': user_data['week'].max(),
                    'last_weight_change': y[-1] if len(y) > 0 else 0
                }
            except:
                pass
    
    def predict_weight_change_7days(
        self,
        health_profile: HealthProfile,
        daily_calorie_target: float,
        user_id: int = None
    ) -> Dict:
        """
        Predict weight change after 7 days (1 week).
        
        Uses physics-based calculation:
        - 1 kg body weight ≈ 7700 calories
        - Caloric deficit/surplus determines weight loss/gain
        
        Args:
            health_profile: User's current health profile
            daily_calorie_target: Target daily calorie intake
            user_id: Optional user ID for personalized model
        
        Returns:
            Dictionary with prediction details
        """
        # Base calculation: caloric balance
        current_daily_need = health_profile.daily_calorie_need
        daily_deficit = current_daily_need - daily_calorie_target
        
        # Weekly balance (7 days)
        weekly_balance = daily_deficit * 7
        
        # Convert to weight (1 kg ≈ 7700 calories, but typically use 7000 for net loss)
        weight_change_kg = weekly_balance / 7000
        
        # Adjust based on BMI category for more realistic prediction
        bmi_adjustment_factors = {
            'Underweight': 0.5,    # Slower weight change for underweight
            'Normal': 0.8,         # Normal metabolic efficiency
            'Overweight': 1.0,     # Standard metabolic efficiency
            'Obese': 1.1           # Slightly faster changes (metabolic adaptation)
        }
        
        adjustment = bmi_adjustment_factors.get(health_profile.bmi_category, 1.0)
        weight_change_kg *= adjustment
        
        # If user has historical data, use ML model for refinement
        if user_id and user_id in self.weight_models:
            try:
                model_info = self.weight_models[user_id]
                # Future week prediction
                future_week = model_info['last_week'] + 1
                X_future = np.array([[future_week, daily_calorie_target]])
                X_future_scaled = model_info['scaler'].transform(X_future)
                ml_prediction = model_info['model'].predict(X_future_scaled)[0]
                
                # Blend physics-based and ML prediction (70% physics, 30% ML)
                weight_change_kg = (weight_change_kg * 0.7) + (ml_prediction * 0.3)
            except:
                pass
        
        # Predict new weight
        predicted_weight = health_profile.weight_kg + weight_change_kg
        predicted_bmi = health_profile.weight_kg / ((health_profile.height_cm / 100) ** 2)
        
        # Determine direction
        if weight_change_kg > 0.1:
            direction = "📈 Weight increase"
        elif weight_change_kg < -0.1:
            direction = "📉 Weight decrease"
        else:
            direction = "➡️ Weight stable"
        
        return {
            'current_weight_kg': health_profile.weight_kg,
            'predicted_weight_kg': round(predicted_weight, 2),
            'weight_change_kg': round(weight_change_kg, 2),
            'direction': direction,
            'current_daily_need': round(current_daily_need, 0),
            'target_daily_intake': round(daily_calorie_target, 0),
            'daily_deficit_kcal': round(daily_deficit, 0),
            'prediction_confidence': 'Medium' if user_id in self.weight_models else 'Moderate'
        }
    
    def predict_food_category_preference(
        self,
        health_profile: HealthProfile,
        recent_foods_data: pd.DataFrame = None
    ) -> Dict:
        """
        Predict next week's most suitable food category based on:
        - BMI trend
        - Activity level
        - Historical preferences (if available)
        
        Args:
            health_profile: User's health profile
            recent_foods_data: Optional DataFrame of foods user recently consumed
        
        Returns:
            Dictionary with category predictions
        """
        # Base category recommendations by BMI and activity
        category_preferences = {
            'Underweight': {
                'High Priority': ['Protein', 'Nuts/Seeds'],
                'Secondary': ['Grains', 'Dairy'],
                'Description': 'Focus on calorie-dense, high-protein options for healthy weight gain'
            },
            'Normal': {
                'High Priority': ['Vegetables', 'Fruits'],
                'Secondary': ['Protein', 'Grains'],
                'Description': 'Maintain balanced nutrition with whole foods'
            },
            'Overweight': {
                'High Priority': ['Vegetables', 'Fruits'],
                'Secondary': ['Protein', 'Grains (whole)'],
                'Description': 'Emphasize low-calorie, high-fiber foods for gradual weight loss'
            },
            'Obese': {
                'High Priority': ['Vegetables', 'Fruits'],
                'Secondary': ['Protein (lean)'],
                'Description': 'Prioritize nutrient-dense, low-calorie foods'
            }
        }
        
        prediction = category_preferences[health_profile.bmi_category]
        
        # Adjust based on activity level
        if health_profile.activity_level == 'Very Active':
            if health_profile.bmi_category != 'Underweight':
                # Active users need more protein
                if 'Protein' not in prediction['High Priority']:
                    prediction['High Priority'] = list(prediction['High Priority']) + ['Protein']
        
        return {
            'bmi_category': health_profile.bmi_category,
            'activity_level': health_profile.activity_level,
            'primary_categories': prediction['High Priority'],
            'secondary_categories': prediction['Secondary'],
            'guidance': prediction['Description'],
            'prediction_basis': 'BMI Category & Activity Level Analysis'
        }
    
    def predict_caloric_needs_adjustment(
        self,
        health_profile: HealthProfile,
        weeks_elapsed: int = 4
    ) -> Dict:
        """
        Predict future caloric needs after weight change.
        BMR changes with weight, so this recalculates as weight changes.
        
        Args:
            health_profile: Current health profile
            weeks_elapsed: Number of weeks in the prediction window
        
        Returns:
            Adjusted caloric needs projection
        """
        from health_calculator import HealthCalculator
        
        predictions = []
        current_weight = health_profile.weight_kg
        
        for week in range(1, weeks_elapsed + 1):
            # Simulate weight change (rough estimate: 0.25-0.5 kg per week on calorie deficit)
            if health_profile.bmi_category in ['Overweight', 'Obese']:
                weekly_loss = 0.3
                current_weight -= weekly_loss
            elif health_profile.bmi_category == 'Underweight':
                weekly_gain = 0.2
                current_weight += weekly_gain
            
            # Recalculate caloric needs at new weight
            new_bmr = HealthCalculator.estimate_bmr(
                health_profile.age,
                health_profile.gender,
                current_weight,
                health_profile.height_cm
            )
            
            new_tdee = HealthCalculator.calculate_daily_calorie_need(
                health_profile.age,
                health_profile.gender,
                current_weight,
                health_profile.height_cm,
                health_profile.activity_level
            )
            
            predictions.append({
                'week': week,
                'projected_weight': round(current_weight, 2),
                'projected_bmr': round(new_bmr, 1),
                'projected_tdee': round(new_tdee, 1)
            })
        
        return {
            'weeks_projected': weeks_elapsed,
            'projections': predictions,
            'trend': 'Decreasing needs' if health_profile.bmi_category in ['Overweight', 'Obese'] else 'Increasing needs'
        }

def print_weight_prediction(prediction: Dict):
    """
    Pretty print weight prediction results.
    
    Args:
        prediction: Dictionary with prediction data
    """
    print("\n" + "="*70)
    print("⏰ 7-DAY WEIGHT PREDICTION")
    print("="*70)
    print(f"Current Weight: {prediction['current_weight_kg']} kg")
    print(f"Predicted Weight: {prediction['predicted_weight_kg']} kg")
    print(f"Weight Change: {prediction['weight_change_kg']} kg {prediction['direction']}")
    print(f"\nCalorie Analysis:")
    print(f"  Daily Need (TDEE): {prediction['current_daily_need']:.0f} kcal")
    print(f"  Target Intake: {prediction['target_daily_intake']:.0f} kcal")
    print(f"  Daily Deficit: {prediction['daily_deficit_kcal']:.0f} kcal")
    print(f"\nPrediction Confidence: {prediction['prediction_confidence']}")
    print("="*70 + "\n")

def print_food_category_prediction(prediction: Dict):
    """
    Pretty print food category prediction.
    
    Args:
        prediction: Dictionary with prediction data
    """
    print("\n" + "="*70)
    print("📊 NEXT WEEK'S FOOD CATEGORY PREDICTION")
    print("="*70)
    print(f"User Category: {prediction['bmi_category']}")
    print(f"Activity Level: {prediction['activity_level']}")
    print(f"\nPrimary Categories: {', '.join(prediction['primary_categories'])}")
    print(f"Secondary Categories: {', '.join(prediction['secondary_categories'])}")
    print(f"\n💡 Guidance: {prediction['guidance']}")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("🏥 HealthVend - Prediction Models")
    print("-" * 70)
    
    from data.generate_datasets import generate_user_progress_dataset, generate_users_dataset
    from health_calculator import HealthCalculator
    
    # Load data
    users_df = generate_users_dataset(50)
    progress_df = generate_user_progress_dataset(50, 4)
    
    # Create prediction model
    predictor = PredictionModel(progress_df, users_df)
    
    # Test prediction
    profile = HealthCalculator.create_health_profile(
        user_id=1,
        age=40,
        gender='M',
        height_cm=178,
        weight_kg=92,
        activity_level='Light'
    )
    
    # Predict weight
    target_calories = 2000
    weight_pred = predictor.predict_weight_change_7days(profile, target_calories)
    print_weight_prediction(weight_pred)
    
    # Predict food categories
    category_pred = predictor.predict_food_category_preference(profile)
    print_food_category_prediction(category_pred)
    
    # Predict caloric needs adjustment
    calorie_adjust = predictor.predict_caloric_needs_adjustment(profile, weeks_elapsed=4)
    print(f"Projected Caloric Needs (Next 4 weeks):")
    for proj in calorie_adjust['projections']:
        print(f"  Week {proj['week']}: {proj['projected_weight']} kg | "
              f"TDEE: {proj['projected_tdee']:.0f} kcal")
