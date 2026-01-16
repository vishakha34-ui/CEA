"""
Health Calculator Module for HealthVend System
Implements BMI calculation and health classification
Part of User Analysis Agent in Agentic AI Architecture
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class HealthProfile:
    """
    Data class representing user's health profile
    Supports the agentic system's need for clear health data structures
    """
    user_id: int
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    bmi: float
    bmi_category: str
    activity_level: str
    daily_calorie_need: float
    
    def to_dict(self) -> Dict:
        """Convert health profile to dictionary"""
        return {
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender,
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'bmi': self.bmi,
            'bmi_category': self.bmi_category,
            'activity_level': self.activity_level,
            'daily_calorie_need': self.daily_calorie_need
        }

class HealthCalculator:
    """
    Calculates BMI, classifies health status, and estimates daily caloric needs.
    
    Academic Context:
    This module supports the "User Analysis Agent" component of the Agentic AI system.
    It provides personalized health assessment enabling:
    - Autonomous health data processing
    - Evidence-based nutritional recommendations
    - Public health service delivery optimization
    """
    
    # Activity level multipliers (Harris-Benedict Formula)
    ACTIVITY_MULTIPLIERS = {
        'Sedentary': 1.2,           # Little to no exercise
        'Light': 1.375,             # 1-3 days/week
        'Moderate': 1.55,           # 3-5 days/week
        'Very Active': 1.725        # 6-7 days/week
    }
    
    # BMI Classification boundaries
    BMI_CLASSIFICATIONS = {
        'Underweight': (0, 18.5),
        'Normal': (18.5, 25),
        'Overweight': (25, 30),
        'Obese': (30, float('inf'))
    }
    
    @staticmethod
    def calculate_bmi(height_cm: float, weight_kg: float) -> float:
        """
        Calculate BMI using standard formula: weight(kg) / height(m)^2
        
        Args:
            height_cm: Height in centimeters
            weight_kg: Weight in kilograms
        
        Returns:
            BMI value rounded to 2 decimal places
        
        Raises:
            ValueError: If inputs are invalid
        """
        if height_cm <= 0 or weight_kg <= 0:
            raise ValueError("Height and weight must be positive values")
        
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    
    @staticmethod
    def classify_bmi(bmi: float) -> str:
        """
        Classify BMI into standard health categories.
        
        Args:
            bmi: BMI value
        
        Returns:
            BMI classification string
        """
        for category, (lower, upper) in HealthCalculator.BMI_CLASSIFICATIONS.items():
            if lower <= bmi < upper:
                return category
        return 'Obese'
    
    @staticmethod
    def estimate_bmr(age: int, gender: str, weight_kg: float, height_cm: float) -> float:
        """
        Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.
        More accurate than Harris-Benedict for modern populations.
        
        Args:
            age: Age in years
            gender: 'M' for male, 'F' for female
            weight_kg: Weight in kilograms
            height_cm: Height in centimeters
        
        Returns:
            BMR in calories per day
        """
        if gender.upper() == 'M':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:  # Female
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        
        return round(bmr, 1)
    
    @staticmethod
    def calculate_daily_calorie_need(
        age: int,
        gender: str,
        weight_kg: float,
        height_cm: float,
        activity_level: str
    ) -> float:
        """
        Calculate total daily energy expenditure (TDEE).
        
        Args:
            age: Age in years
            gender: 'M' for male, 'F' for female
            weight_kg: Weight in kilograms
            height_cm: Height in centimeters
            activity_level: One of 'Sedentary', 'Light', 'Moderate', 'Very Active'
        
        Returns:
            Daily calorie need in calories
        """
        bmr = HealthCalculator.estimate_bmr(age, gender, weight_kg, height_cm)
        multiplier = HealthCalculator.ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
        tdee = bmr * multiplier
        return round(tdee, 1)
    
    @staticmethod
    def create_health_profile(
        user_id: int,
        age: int,
        gender: str,
        height_cm: float,
        weight_kg: float,
        activity_level: str
    ) -> HealthProfile:
        """
        Create a comprehensive health profile for a user.
        
        Args:
            user_id: User identifier
            age: Age in years
            gender: 'M' or 'F'
            height_cm: Height in centimeters
            weight_kg: Weight in kilograms
            activity_level: Activity level string
        
        Returns:
            HealthProfile object with all calculated metrics
        """
        bmi = HealthCalculator.calculate_bmi(height_cm, weight_kg)
        bmi_category = HealthCalculator.classify_bmi(bmi)
        daily_calorie_need = HealthCalculator.calculate_daily_calorie_need(
            age, gender, weight_kg, height_cm, activity_level
        )
        
        return HealthProfile(
            user_id=user_id,
            age=age,
            gender=gender,
            height_cm=height_cm,
            weight_kg=weight_kg,
            bmi=bmi,
            bmi_category=bmi_category,
            activity_level=activity_level,
            daily_calorie_need=daily_calorie_need
        )
    
    @staticmethod
    def get_calorie_deficit_target(
        bmi_category: str,
        daily_calorie_need: float
    ) -> float:
        """
        Calculate target calorie intake based on BMI category.
        Provides personalized nutrition guidance.
        
        Args:
            bmi_category: BMI classification
            daily_calorie_need: Calculated TDEE
        
        Returns:
            Target daily calorie intake
        """
        # For weight loss: 15-20% deficit; maintenance: no deficit; weight gain: 10-15% surplus
        if bmi_category == 'Underweight':
            target = daily_calorie_need * 1.10  # 10% surplus for healthy weight gain
        elif bmi_category == 'Normal':
            target = daily_calorie_need  # Maintenance calories
        elif bmi_category == 'Overweight':
            target = daily_calorie_need * 0.85  # 15% deficit for gradual weight loss
        else:  # Obese
            target = daily_calorie_need * 0.80  # 20% deficit for sustainable weight loss
        
        return round(target, 1)

def print_health_profile(profile: HealthProfile):
    """
    Pretty print a health profile.
    
    Args:
        profile: HealthProfile object to display
    """
    print("\n" + "="*60)
    print("📊 HEALTH PROFILE ANALYSIS")
    print("="*60)
    print(f"User ID: {profile.user_id}")
    print(f"Age: {profile.age} years | Gender: {profile.gender}")
    print(f"Height: {profile.height_cm} cm | Weight: {profile.weight_kg} kg")
    print(f"BMI: {profile.bmi} ({profile.bmi_category})")
    print(f"Activity Level: {profile.activity_level}")
    print(f"Daily Calorie Need: {profile.daily_calorie_need:.0f} kcal")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Example usage
    print("🏥 HealthVend - Health Calculator Module")
    print("-" * 60)
    
    # Create sample health profiles
    profile1 = HealthCalculator.create_health_profile(
        user_id=1,
        age=30,
        gender='M',
        height_cm=175,
        weight_kg=85,
        activity_level='Moderate'
    )
    
    profile2 = HealthCalculator.create_health_profile(
        user_id=2,
        age=28,
        gender='F',
        height_cm=165,
        weight_kg=60,
        activity_level='Light'
    )
    
    print_health_profile(profile1)
    print_health_profile(profile2)
    
    # Show calorie targets
    print("📋 CALORIE TARGET RECOMMENDATIONS")
    print("-" * 60)
    for profile in [profile1, profile2]:
        target = HealthCalculator.get_calorie_deficit_target(
            profile.bmi_category,
            profile.daily_calorie_need
        )
        print(f"User {profile.user_id} ({profile.bmi_category}): {target:.0f} kcal/day")
