"""
HealthVend - AI-Enabled Smart Vending Food Recommendation System
Agentic AI-Based Intelligent Public Service Assistance System
For hospitals, universities, gyms, and public health environments
"""

import sys
import os
import pandas as pd
from typing import Dict, List

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from data.generate_datasets import save_datasets
from health_calculator import HealthCalculator, print_health_profile
from recommendation_engine import RecommendationEngine, print_recommendations
from prediction_model import PredictionModel, print_weight_prediction, print_food_category_prediction
from agents.user_agent import UserAnalysisAgent
from agents.nutrition_agent import NutritionIntelligenceAgent
from agents.prediction_agent import PredictionAgent
from agents.learning_agent import LearningAgent


class HealthVendSystem:
    """
    Main HealthVend System Orchestrator
    
    Coordinates autonomous agents for integrated health service delivery.
    
    ACADEMIC CONTEXT:
    This system implements an "Agentic AI–Based Intelligent Public Service Assistance System"
    with the following characteristics:
    
    1. AUTONOMY: Each agent operates independently within defined responsibilities
    2. PERSONALIZATION: Recommendations tailored to individual health profiles
    3. SCALABILITY: Architecture supports deployment across multiple institutions
    4. PUBLIC HEALTH IMPACT: Designed for hospitals, universities, gyms, and wellness centers
    5. EVIDENCE-BASED: Grounded in nutritional science and health guidelines
    6. CONTINUOUS LEARNING: System improves through historical data analysis
    
    AGENTS:
    - User Analysis Agent: Health profile creation and classification
    - Nutrition Intelligence Agent: Personalized food recommendations
    - Prediction Agent: Weight and nutrition forecasting
    - Learning Agent: System improvement and pattern recognition
    """
    
    def __init__(self):
        """Initialize HealthVend System"""
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.users_df = None
        self.foods_df = None
        self.progress_df = None
        
        # Initialize agents
        self.user_agent = None
        self.nutrition_agent = None
        self.prediction_agent = None
        self.learning_agent = None
        
        print("\n" + "="*80)
        print("HEALTHVEND - AGENTIC AI-BASED HEALTH RECOMMENDATION SYSTEM")
        print("="*80)
        print("\nSystem Features:")
        print("  * BMI Calculation & Health Classification")
        print("  * Hybrid ML + Rule-Based Food Recommendations")
        print("  * Weight & Nutrition Forecasting")
        print("  * Multi-Agent Autonomous Architecture")
        print("  * Scalable Public Health Service Delivery")
        print()
    
    def initialize_system(self):
        """Initialize and load all system components"""
        print("\nSYSTEM INITIALIZATION")
        print("-" * 80)
        
        # Generate/load datasets
        self._load_datasets()
        
        # Initialize agents
        self._initialize_agents()
        
        print("✅ System initialization complete!\n")
    
    def _load_datasets(self):
        """Load or generate datasets"""
        try:
            # Check if datasets exist
            users_path = os.path.join(self.data_dir, 'users.csv')
            foods_path = os.path.join(self.data_dir, 'foods.csv')
            progress_path = os.path.join(self.data_dir, 'user_progress.csv')
            
            if os.path.exists(users_path) and os.path.exists(foods_path):
                print("Loading existing datasets...")
                self.users_df = pd.read_csv(users_path)
                self.foods_df = pd.read_csv(foods_path)
                self.progress_df = pd.read_csv(progress_path) if os.path.exists(progress_path) else None
            else:
                print("Generating synthetic datasets...")
                self.users_df, self.foods_df, self.progress_df = save_datasets(self.data_dir)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("Generating new datasets...")
            self.users_df, self.foods_df, self.progress_df = save_datasets(self.data_dir)
    
    def _initialize_agents(self):
        """Initialize all autonomous agents"""
        print("Initializing autonomous agents...")
        
        self.user_agent = UserAnalysisAgent()
        print("  [OK] User Analysis Agent ready")
        
        self.nutrition_agent = NutritionIntelligenceAgent(self.foods_df, self.users_df)
        print("  [OK] Nutrition Intelligence Agent ready")
        
        if self.progress_df is not None:
            self.prediction_agent = PredictionAgent(self.progress_df, self.users_df)
            self.learning_agent = LearningAgent(self.progress_df, self.users_df)
            print("  [OK] Prediction Agent ready")
            print("  [OK] Learning Agent ready")
    
    def run_demo(self):
        """Run comprehensive system demonstration"""
        print("\n" + "="*80)
        print("HEALTHVEND SYSTEM DEMONSTRATION")
        print("="*80)
        
        # Demo user inputs
        self.demo_users = [
            {
                'user_id': 101,
                'age': 35,
                'gender': 'M',
                'height_cm': 180,
                'weight_kg': 95,
                'activity_level': 'Moderate',
                'name': 'John (Overweight)'
            },
            {
                'user_id': 102,
                'age': 28,
                'gender': 'F',
                'height_cm': 165,
                'weight_kg': 58,
                'activity_level': 'Light',
                'name': 'Sarah (Underweight)'
            },
            {
                'user_id': 103,
                'age': 42,
                'gender': 'M',
                'height_cm': 172,
                'weight_kg': 105,
                'activity_level': 'Sedentary',
                'name': 'Mike (Obese)'
            }
        ]
        
        for user_info in self.demo_users:
            self._process_user(user_info)
    
    def _process_user(self, user_info: Dict):
        """Process a single user through all agents"""
        print("\n" + "-"*80)
        print(f"USER: {user_info['name']}")
        print("-"*80)
        
        # AGENT 1: User Analysis Agent
        # Purpose: Create health profile and perform initial assessment
        print("\n[AGENT 1] USER ANALYSIS AGENT - Health Assessment")
        print("-" * 80)
        
        profile = self.user_agent.collect_user_input(
            user_id=user_info['user_id'],
            age=user_info['age'],
            gender=user_info['gender'],
            height_cm=user_info['height_cm'],
            weight_kg=user_info['weight_kg'],
            activity_level=user_info['activity_level']
        )
        
        print_health_profile(profile)
        
        analysis = self.user_agent.analyze_health_status(profile)
        print("Health Analysis Summary:")
        print(f"  BMI Classification: {analysis['health_metrics']['bmi_category']}")
        print(f"  Daily Calorie Need (TDEE): {analysis['health_metrics']['tdee_kcal']:.0f} kcal")
        print(f"  Target Daily Intake: {analysis['health_metrics']['target_daily_intake']:.0f} kcal")
        print(f"\nAssessment: {analysis['health_assessment']}")
        print(f"\nPriority Actions:")
        for i, action in enumerate(analysis['priority_actions'], 1):
            print(f"  {i}. {action}")
        
        # AGENT 2: Nutrition Intelligence Agent
        # Purpose: Generate personalized food recommendations
        print("\n[AGENT 2] NUTRITION INTELLIGENCE AGENT - Food Recommendations")
        print("-" * 80)
        
        nutrition_result = self.nutrition_agent.generate_recommendations(
            user_info['user_id'],
            profile,
            num_recommendations=3
        )
        
        print_recommendations(nutrition_result['recommendations'])
        
        print("Nutritional Summary of Recommendations:")
        summary = nutrition_result['nutritional_summary']
        print(f"  Total Calories: {summary['total_calories']} kcal")
        print(f"  Protein: {summary['total_protein_g']}g ({summary['protein_percentage']:.1f}%)")
        print(f"  Fat: {summary['total_fat_g']}g ({summary['fat_percentage']:.1f}%)")
        print(f"  Carbs: {summary['total_carbs_g']}g ({summary['carbs_percentage']:.1f}%)")
        print(f"  Fiber: {summary['total_fiber_g']}g")
        
        print(f"\nMacronutrient Balance: {nutrition_result['adequacy_assessment']['macronutrient_balance']['assessment']}")
        
        # AGENT 3: Prediction Agent
        # Purpose: Forecast weight changes and future needs
        print("\n[AGENT 3] PREDICTION AGENT - Future Health Projections")
        print("-" * 80)
        
        target_calories = nutrition_result['nutritional_summary']['total_calories']
        
        weight_forecast = self.prediction_agent.forecast_weight_outcome(
            user_info['user_id'],
            profile,
            target_calories
        )
        
        print_weight_prediction(weight_forecast['weight_prediction'])
        
        print("\nAdjustment Recommendations:")
        for rec in weight_forecast['recommendations']:
            print(f"  {rec}")
        
        # Food category prediction
        food_forecast = self.prediction_agent.forecast_food_needs(user_info['user_id'], profile)
        print(f"\nNext Week Food Category Preferences:")
        print(f"  Primary: {', '.join(food_forecast['prediction']['primary_categories'])}")
        print(f"  Secondary: {', '.join(food_forecast['prediction']['secondary_categories'])}")
        
        # AGENT 4: Learning Agent (Population-level insights)
        # Purpose: Extract insights and recommend improvements
        if self.learning_agent and user_info['user_id'] == self.demo_users[-1]['user_id']:
            print("\n" + "="*80)
            print("[AGENT 4] LEARNING AGENT - System Insights & Improvements")
            print("="*80)
            
            # Population patterns
            patterns = self.learning_agent.identify_population_patterns()
            
            print(f"\nPopulation Analysis (across {patterns['total_users']} users):")
            print("\nPerformance by BMI Category:")
            for bmi_cat, perf in patterns['bmi_category_analysis'].items():
                print(f"  {bmi_cat}:")
                print(f"    - Avg Weight Change: {perf['avg_weight_change_kg']} kg")
                print(f"    - Success Rate: {perf['avg_success_rate']}%")
                print(f"    - User Count: {perf['user_count']}")
            
            # System recommendations
            improvements = self.learning_agent.recommend_system_improvements()
            print(f"\nSystem Improvement Recommendations:")
            for i, rec in enumerate(improvements[:3], 1):
                print(f"\n  {i}. [{rec['priority']}] {rec['area']}")
                print(f"     Recommendation: {rec['recommendation']}")
                print(f"     Impact: {rec['impact']}")
        
        print("\n" + "="*80 + "\n")
    
    def get_user_recommendations(self, user_id: int, age: int, gender: str, 
                                height_cm: float, weight_kg: float, 
                                activity_level: str) -> Dict:
        """
        Public API method for getting recommendations for a user
        
        Args:
            user_id: User identifier
            age: Age in years
            gender: 'M' or 'F'
            height_cm: Height in centimeters
            weight_kg: Weight in kilograms
            activity_level: Activity level classification
        
        Returns:
            Dictionary with complete user recommendations
        """
        # Create health profile
        profile = self.user_agent.collect_user_input(
            user_id, age, gender, height_cm, weight_kg, activity_level
        )
        
        # Get nutrition recommendations
        nutrition_result = self.nutrition_agent.generate_recommendations(
            user_id, profile, num_recommendations=3
        )
        
        # Get predictions
        weight_forecast = self.prediction_agent.forecast_weight_outcome(
            user_id, profile, nutrition_result['nutritional_summary']['total_calories']
        )
        
        return {
            'health_profile': profile.to_dict(),
            'recommendations': nutrition_result['recommendations'],
            'nutritional_summary': nutrition_result['nutritional_summary'],
            'weight_forecast': weight_forecast['weight_prediction']
        }


def main():
    """Main entry point"""
    try:
        # Initialize system
        system = HealthVendSystem()
        system.initialize_system()
        
        # Run demonstration
        system.run_demo()
        
        # Summary
        print("\n" + "="*80)
        print("HEALTHVEND SYSTEM DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nKey System Capabilities Demonstrated:")
        print("  1. [OK] BMI Calculation & Health Classification")
        print("  2. [OK] Hybrid ML + Rule-Based Food Recommendations")
        print("  3. [OK] Personalized Nutritional Analysis")
        print("  4. [OK] Weight Change Prediction (7-day forecast)")
        print("  5. [OK] Food Category Preference Forecasting")
        print("  6. [OK] Multi-Agent Autonomous Architecture")
        print("  7. [OK] Population-Level Learning & Insights")
        
        print("\nDEPLOYMENT CONTEXTS:")
        print("  * Hospitals: Patient nutrition management & dietary planning")
        print("  * Universities: Campus dining personalization & nutrition education")
        print("  * Gyms/Wellness: Fitness-aligned nutrition recommendations")
        print("  * Public Health: Population health monitoring & intervention")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
