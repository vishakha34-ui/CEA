"""
Recommendation Engine for HealthVend System
Hybrid approach combining rule-based filtering and Machine Learning recommendations
Part of Nutrition Intelligence Agent in Agentic AI Architecture
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from health_calculator import HealthProfile, HealthCalculator

class RecommendationEngine:
    """
    Provides personalized food recommendations using hybrid approach:
    1. Rule-based filtering (BMI suitability, calorie limits)
    2. Machine Learning classification (Decision Tree)
    
    Academic Context:
    This module supports the "Nutrition Intelligence Agent" in the Agentic AI system.
    It delivers:
    - Autonomous food selection based on health metrics
    - Personalized nutritional guidance
    - Scalable public health service delivery
    - Evidence-based recommendations for hospitals, universities, and gyms
    """
    
    def __init__(self, foods_df: pd.DataFrame, users_df: pd.DataFrame):
        """
        Initialize recommendation engine.
        
        Args:
            foods_df: DataFrame with food items and nutritional data
            users_df: DataFrame with historical user data
        """
        self.foods_df = foods_df.copy()
        self.users_df = users_df.copy()
        self.ml_model = None
        self.label_encoder = LabelEncoder()
        self._prepare_ml_model()
    
    def _prepare_ml_model(self):
        """
        Prepare Machine Learning model (Decision Tree) for food recommendations.
        Trains on user BMI categories and their optimal food categories.
        """
        try:
            # Create training data mapping BMI categories to preferred food categories
            # Based on nutritional guidelines
            training_data = []
            bmi_to_category = {
                'Underweight': ['Nuts/Seeds', 'Dairy', 'Protein', 'Grains'],
                'Normal': ['Vegetables', 'Fruits', 'Protein', 'Grains', 'Dairy'],
                'Overweight': ['Vegetables', 'Fruits', 'Protein'],
                'Obese': ['Vegetables', 'Fruits', 'Protein']
            }
            
            for bmi_cat, food_cats in bmi_to_category.items():
                for food_cat in food_cats:
                    training_data.append({
                        'bmi_category': bmi_cat,
                        'recommended_category': food_cat
                    })
            
            training_df = pd.DataFrame(training_data)
            
            # Encode categorical variables
            X = self.label_encoder.fit_transform(training_df['bmi_category']).reshape(-1, 1)
            y = training_df['recommended_category']
            
            # Train Decision Tree
            self.ml_model = DecisionTreeClassifier(max_depth=3, random_state=42)
            self.ml_model.fit(X, y)
            
        except Exception as e:
            print(f"⚠️ ML model training warning: {e}")
            self.ml_model = None
    
    def rule_based_filter(
        self,
        health_profile: HealthProfile,
        calories_available: float = None
    ) -> pd.DataFrame:
        """
        Rule-based filtering: Find foods suitable for user's BMI category.
        
        Args:
            health_profile: User's health profile
            calories_available: Available calories for recommendation (optional)
        
        Returns:
            Filtered DataFrame of suitable foods
        """
        # Filter by BMI suitability
        suitable_foods = self.foods_df[
            (self.foods_df['bmi_suitability'] == health_profile.bmi_category) |
            (self.foods_df['bmi_suitability'] == 'All')
        ].copy()
        
        # Filter by calorie limit if provided
        if calories_available is None:
            target = HealthCalculator.get_calorie_deficit_target(
                health_profile.bmi_category,
                health_profile.daily_calorie_need
            )
            # Average recommended calories per meal (assuming 3 meals + 1-2 snacks)
            calories_available = target / 5
        
        suitable_foods = suitable_foods[suitable_foods['calories'] <= calories_available * 1.2]
        
        return suitable_foods
    
    def calculate_nutrition_score(self, food_row: pd.Series, profile: HealthProfile) -> float:
        """
        Calculate a nutrition score for a food item based on user profile.
        
        Scoring factors:
        - High protein: +bonus (essential for all, especially important for muscle)
        - High fiber: +bonus (helps with satiety and digestion)
        - Low fat: +bonus for overweight/obese
        - Balanced macros: +bonus
        
        Args:
            food_row: Food item row from DataFrame
            profile: User's health profile
        
        Returns:
            Nutrition score (higher is better)
        """
        score = 0.0
        
        # Protein quality (20-40g per serving is optimal)
        protein = food_row['protein_g']
        if protein >= 15:
            score += 30
        elif protein >= 10:
            score += 20
        elif protein >= 5:
            score += 10
        
        # Fiber quality (3-8g is good)
        fiber = food_row['fiber_g']
        if fiber >= 5:
            score += 20
        elif fiber >= 3:
            score += 10
        
        # Fat consideration (lower is better for overweight/obese)
        fat = food_row['fat_g']
        if profile.bmi_category in ['Overweight', 'Obese']:
            if fat <= 5:
                score += 15
            elif fat <= 10:
                score += 10
        else:
            if fat <= 10:
                score += 10
        
        # Carbs to fiber ratio (lower ratio is better)
        if food_row['carbs_g'] > 0:
            carb_fiber_ratio = food_row['carbs_g'] / (food_row['fiber_g'] + 1)
            if carb_fiber_ratio <= 5:
                score += 15
            elif carb_fiber_ratio <= 8:
                score += 10
        
        # Calorie efficiency for overweight/obese
        if profile.bmi_category in ['Overweight', 'Obese']:
            if food_row['calories'] <= 200:
                score += 10
        
        return score
    
    def recommend_foods(
        self,
        health_profile: HealthProfile,
        top_n: int = 3,
        use_ml: bool = True
    ) -> List[Dict]:
        """
        Generate top food recommendations for a user.
        
        Combines:
        1. Rule-based filtering (BMI + calories)
        2. ML-based category suggestion
        3. Nutrition scoring
        
        Args:
            health_profile: User's health profile
            top_n: Number of recommendations to return
            use_ml: Whether to use ML model for category suggestion
        
        Returns:
            List of recommended foods with scores and reasons
        """
        # Step 1: Rule-based filtering
        suitable_foods = self.rule_based_filter(health_profile)
        
        if suitable_foods.empty:
            # Fallback to all foods if no suitable found
            suitable_foods = self.foods_df.copy()
        
        # Step 2: ML-based category preference
        preferred_categories = []
        if use_ml and self.ml_model:
            try:
                bmi_encoded = self.label_encoder.transform([health_profile.bmi_category])[0]
                preferred_cat = self.ml_model.predict([[bmi_encoded]])[0]
                preferred_categories = [preferred_cat]
                
                # Get more categories from the model's prediction confidence
                if len(self.label_encoder.classes_) > 1:
                    # Add complementary categories
                    if health_profile.bmi_category in ['Normal']:
                        preferred_categories.extend(['Vegetables', 'Fruits', 'Protein'])
                    elif health_profile.bmi_category in ['Overweight', 'Obese']:
                        preferred_categories.extend(['Vegetables', 'Fruits'])
            except:
                pass
        
        # Step 3: Calculate nutrition scores
        suitable_foods['nutrition_score'] = suitable_foods.apply(
            lambda row: self.calculate_nutrition_score(row, health_profile),
            axis=1
        )
        
        # Boost score for ML-preferred categories
        if preferred_categories:
            suitable_foods['nutrition_score'] = suitable_foods.apply(
                lambda row: row['nutrition_score'] + 10 if row['category'] in preferred_categories else row['nutrition_score'],
                axis=1
            )
        
        # Step 4: Sort and get top recommendations
        recommendations = suitable_foods.sort_values('nutrition_score', ascending=False).head(top_n)
        
        # Format recommendations
        result = []
        for idx, (_, food) in enumerate(recommendations.iterrows(), 1):
            reason = self._generate_recommendation_reason(food, health_profile)
            result.append({
                'rank': idx,
                'food_id': food['food_id'],
                'food_name': food['food_name'],
                'category': food['category'],
                'calories': food['calories'],
                'protein_g': food['protein_g'],
                'fat_g': food['fat_g'],
                'carbs_g': food['carbs_g'],
                'fiber_g': food['fiber_g'],
                'nutrition_score': round(food['nutrition_score'], 2),
                'reason': reason
            })
        
        return result
    
    def _generate_recommendation_reason(self, food: pd.Series, profile: HealthProfile) -> str:
        """
        Generate a human-readable reason for recommendation.
        
        Args:
            food: Food item
            profile: User's health profile
        
        Returns:
            Explanation string
        """
        reasons = []
        
        # Protein content
        if food['protein_g'] >= 15:
            reasons.append("high protein")
        
        # Fiber content
        if food['fiber_g'] >= 5:
            reasons.append("high fiber")
        
        # Low calorie for overweight/obese
        if profile.bmi_category in ['Overweight', 'Obese'] and food['calories'] <= 200:
            reasons.append("low calorie")
        
        # Nutritional balance
        if food['protein_g'] > 10 and food['fiber_g'] >= 3:
            reasons.append("balanced nutrition")
        
        # BMI suitability
        if food['bmi_suitability'] == profile.bmi_category:
            reasons.append(f"optimized for {profile.bmi_category}")
        
        if not reasons:
            reasons.append("nutritionally balanced")
        
        return " • ".join(reasons)

def print_recommendations(recommendations: List[Dict]):
    """
    Pretty print food recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
    """
    print("\n" + "="*80)
    print("🍎 TOP FOOD RECOMMENDATIONS")
    print("="*80)
    
    for rec in recommendations:
        print(f"\n#{rec['rank']} - {rec['food_name']} ({rec['category']})")
        print(f"   Score: {rec['nutrition_score']}/100")
        print(f"   Calories: {rec['calories']} | Protein: {rec['protein_g']}g | "
              f"Fat: {rec['fat_g']}g | Carbs: {rec['carbs_g']}g | Fiber: {rec['fiber_g']}g")
        print(f"   ✨ Why: {rec['reason']}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Example usage
    print("🏥 HealthVend - Recommendation Engine")
    print("-" * 80)
    
    # Load datasets
    from data.generate_datasets import generate_users_dataset, generate_foods_dataset
    
    users_df = generate_users_dataset(50)
    foods_df = generate_foods_dataset(40)
    
    # Create engine
    engine = RecommendationEngine(foods_df, users_df)
    
    # Get recommendations for sample user
    from health_calculator import HealthCalculator
    
    profile = HealthCalculator.create_health_profile(
        user_id=1,
        age=35,
        gender='M',
        height_cm=180,
        weight_kg=95,
        activity_level='Moderate'
    )
    
    print(f"\nUser Profile: {profile.bmi_category} (BMI: {profile.bmi})")
    
    recommendations = engine.recommend_foods(profile, top_n=3)
    print_recommendations(recommendations)
