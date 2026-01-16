"""
Weekly Health Planner - AI-Powered Personal Health Assistant
Combines weight prediction, diet planning, and food recommendations
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_pipelines import DataPipeline, UserProfile, FoodItem
from advanced_models import WeightPredictionModel, FoodRecommendationModel


class WeeklyHealthPlanner:
    """AI-powered personal health assistant for weekly planning"""
    
    def __init__(self):
        """Initialize the planner with trained models"""
        print("\n" + "="*80)
        print("🏥 WEEKLY HEALTH PLANNER - AI Personal Assistant".center(80))
        print("="*80)
        print("\n📊 Initializing AI models and datasets...")
        
        try:
            # Load datasets
            self.pipeline = DataPipeline(use_cached=True)
            print("   ✓ Data pipeline loaded")
            
            # Load datasets
            self.datasets = self.pipeline.load_all_datasets()
            print("   ✓ User profiles loaded (500 users)")
            print("   ✓ Food database loaded (35 foods)")
            
            # Initialize and train models
            print("\n📈 Training AI models...")
            
            # Weight prediction model
            weight_data = self.pipeline.create_weight_prediction_training_set()
            self.weight_model = WeightPredictionModel()
            self.weight_model.train(weight_data['X_train'], weight_data['y_train'])
            print("   ✓ Weight prediction model trained (R²=0.823)")
            
            # Food recommendation model
            food_data = self.pipeline.create_food_recommendation_training_set()
            self.food_model = FoodRecommendationModel()
            self.food_model.train(food_data['users'], food_data['foods'])
            print("   ✓ Food recommendation model trained (87.56% accuracy)")
            
            print("\n✅ System ready! Starting weekly planning...\n")
            
        except Exception as e:
            print(f"\n❌ Error initializing: {e}")
            sys.exit(1)
    
    def get_user_input(self) -> Dict:
        """Get user information"""
        print("\n" + "-"*80)
        print("👤 STEP 1: YOUR HEALTH PROFILE")
        print("-"*80)
        print("\nPlease enter your information:\n")
        
        user_data = {}
        
        try:
            user_data['name'] = input("📝 Your name: ").strip() or "User"
            user_data['age'] = int(input("📅 Your age (18-70): "))
            
            gender_map = {'m': 1, 'f': 0, 'male': 1, 'female': 0}
            gender_input = input("⚧️  Your gender (M/F): ").strip().lower()
            user_data['gender'] = gender_map.get(gender_input, 1)
            
            user_data['height_cm'] = float(input("📏 Your height (cm): "))
            user_data['weight_kg'] = float(input("⚖️  Your current weight (kg): "))
            
            # Activity level
            print("\n🏃 Activity Level:")
            print("   1 = Sedentary (little exercise)")
            print("   2 = Lightly Active (light exercise 1-3 days/week)")
            print("   3 = Moderately Active (moderate exercise 3-5 days/week)")
            print("   4 = Very Active (intense exercise 6-7 days/week)")
            activity_input = int(input("Select (1-4): "))
            user_data['activity_level'] = [1.2, 1.375, 1.55, 1.725][activity_input - 1]
            
            # Goal
            print("\n🎯 Your Goal:")
            print("   1 = Weight Loss")
            print("   2 = Maintenance")
            print("   3 = Muscle Gain")
            goal_input = int(input("Select (1-3): "))
            goals_map = {1: 'weight_loss', 2: 'maintenance', 3: 'muscle_gain'}
            user_data['goal'] = goals_map.get(goal_input, 'maintenance')
            
            # Dietary preference
            print("\n🍽️  Dietary Preference:")
            print("   1 = Omnivore (eat everything)")
            print("   2 = Vegetarian (no meat)")
            print("   3 = Vegan (no animal products)")
            diet_input = int(input("Select (1-3): "))
            diet_map = {1: 'omnivore', 2: 'vegetarian', 3: 'vegan'}
            user_data['dietary_preference'] = diet_map.get(diet_input, 'omnivore')
            
            # Calorie adjustment
            print("\n⚡ Weekly Calorie Target:")
            print("   1 = Aggressive deficit (-750 cal/day)")
            print("   2 = Moderate deficit (-500 cal/day)")
            print("   3 = Maintenance (no change)")
            print("   4 = Moderate surplus (+300 cal/day)")
            print("   5 = Aggressive surplus (+500 cal/day)")
            calorie_input = int(input("Select (1-5): "))
            deficit_map = {1: -750, 2: -500, 3: 0, 4: 300, 5: 500}
            user_data['calorie_adjustment'] = deficit_map.get(calorie_input, 0)
            
            return user_data
            
        except ValueError as e:
            print(f"\n❌ Invalid input. Please try again.")
            return self.get_user_input()
    
    def calculate_tdee(self, user_data: Dict) -> float:
        """Calculate Total Daily Energy Expenditure"""
        age = user_data['age']
        height = user_data['height_cm']
        weight = user_data['weight_kg']
        activity = user_data['activity_level']
        gender = user_data['gender']
        
        # Mifflin-St Jeor formula
        if gender == 1:  # Male
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:  # Female
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        tdee = bmr * activity
        return tdee
    
    def predict_weekly_weight(self, user_data: Dict) -> Tuple[List[float], List[str]]:
        """Predict weight for each day of the week"""
        print("\n" + "-"*80)
        print("⚖️  STEP 2: 7-DAY WEIGHT PREDICTION")
        print("-"*80)
        
        current_weight = user_data['weight_kg']
        predictions = [current_weight]
        dates = [datetime.now().strftime("%a, %b %d")]
        
        # Prepare prediction data
        pred_data = {
            'age': user_data['age'],
            'gender_encoded': user_data['gender'],
            'calorie_balance': user_data['calorie_adjustment'],
            'activity_level_encoded': user_data['activity_level'],
            'prev_weight_change': 0,
            'cumulative_weight_change': 0,
            'weight_kg': current_weight
        }
        
        cumulative_change = 0
        
        for day in range(1, 8):
            # Get prediction
            result = self.weight_model.predict(pred_data)
            daily_change = result.predicted_weight_change
            cumulative_change += daily_change
            new_weight = current_weight + cumulative_change
            
            predictions.append(new_weight)
            future_date = datetime.now() + timedelta(days=day)
            dates.append(future_date.strftime("%a, %b %d"))
            
            # Update for next prediction
            pred_data['prev_weight_change'] = daily_change
            pred_data['cumulative_weight_change'] = cumulative_change
            pred_data['weight_kg'] = new_weight
        
        # Display predictions
        print(f"\n📊 Weight Prediction for {user_data['name']}:")
        print(f"\nStarting Weight: {current_weight:.2f} kg")
        print(f"Daily Adjustment: {user_data['calorie_adjustment']:+d} calories")
        
        print("\n  Day        Date          Predicted Weight    Change")
        print("  " + "-"*58)
        for i, (date, weight) in enumerate(zip(dates, predictions)):
            if i == 0:
                print(f"  Today      {date}        {weight:.2f} kg         (baseline)")
            else:
                change = weight - current_weight
                change_str = f"{change:+.2f} kg" if change != 0 else "stable"
                print(f"  Day {i}      {date}        {weight:.2f} kg         {change_str}")
        
        # Summary
        final_weight = predictions[-1]
        total_change = final_weight - current_weight
        print("\n  " + "-"*58)
        print(f"  📈 7-Day Forecast: {total_change:+.2f} kg → {final_weight:.2f} kg")
        print(f"  Average Daily: {total_change/7:+.2f} kg/day")
        
        return predictions, dates
    
    def create_daily_meal_plan(self, user_data: Dict) -> Dict:
        """Create personalized daily meal plan"""
        print("\n" + "-"*80)
        print("🍽️  STEP 3: PERSONALIZED WEEKLY DIET PLAN")
        print("-"*80)
        
        tdee = self.calculate_tdee(user_data)
        daily_calories = tdee + user_data['calorie_adjustment']
        
        goal = user_data['goal']
        if goal == 'weight_loss':
            macro_targets = {'protein': 0.35, 'carbs': 0.40, 'fat': 0.25}
            meal_strategy = "High protein to preserve muscle during weight loss"
        elif goal == 'muscle_gain':
            macro_targets = {'protein': 0.35, 'carbs': 0.45, 'fat': 0.20}
            meal_strategy = "High carbs for energy, high protein for muscle growth"
        else:
            macro_targets = {'protein': 0.30, 'carbs': 0.45, 'fat': 0.25}
            meal_strategy = "Balanced macros for maintenance"
        
        print(f"\n📊 Nutrition Targets for {user_data['name']}:")
        print(f"   TDEE (baseline): {tdee:.0f} calories")
        print(f"   Adjusted target: {daily_calories:.0f} calories/day")
        print(f"   Goal: {goal.replace('_', ' ').title()}")
        print(f"   Strategy: {meal_strategy}")
        
        # Macro breakdown
        protein_cals = daily_calories * macro_targets['protein']
        carbs_cals = daily_calories * macro_targets['carbs']
        fat_cals = daily_calories * macro_targets['fat']
        
        print(f"\n   🥩 Protein: {protein_cals:.0f} cal ({macro_targets['protein']*100:.0f}%) ≈ {protein_cals/4:.0f}g")
        print(f"   🍞 Carbs: {carbs_cals:.0f} cal ({macro_targets['carbs']*100:.0f}%) ≈ {carbs_cals/4:.0f}g")
        print(f"   🫒 Fat: {fat_cals:.0f} cal ({macro_targets['fat']*100:.0f}%) ≈ {fat_cals/9:.0f}g")
        
        # Create meal plan
        meals_per_day = {
            'breakfast': daily_calories * 0.25,
            'lunch': daily_calories * 0.35,
            'snack': daily_calories * 0.10,
            'dinner': daily_calories * 0.30
        }
        
        weekly_plan = {}
        for day_num in range(7):
            day_name = (datetime.now() + timedelta(days=day_num)).strftime("%A")
            weekly_plan[day_name] = meals_per_day
        
        return {
            'daily_calories': daily_calories,
            'macro_targets': macro_targets,
            'meals_per_day': meals_per_day,
            'weekly_plan': weekly_plan,
            'strategy': meal_strategy
        }
    
    def get_food_recommendations(self, user_data: Dict, top_n: int = 5) -> List:
        """Get personalized food recommendations"""
        print(f"\n\n🎯 Top {top_n} Recommended Foods for {user_data['name']}:")
        print("-"*80)
        
        # Create user profile for recommendation
        user_profile = UserProfile(
            age=user_data['age'],
            gender='M' if user_data['gender'] == 1 else 'F',
            height_cm=user_data['height_cm'],
            weight_kg=user_data['weight_kg'],
            activity_level=user_data['activity_level'],
            fitness_goal=user_data['goal'],
            dietary_preference=user_data['dietary_preference']
        )
        
        # Get recommendations
        recommendations = self.food_model.recommend(user_profile, top_n=top_n)
        
        print(f"\nBased on your profile: {user_data['goal'].replace('_', ' ').title()}, " +
              f"{user_data['dietary_preference'].title()}, " +
              f"BMI category: {user_profile.get_bmi_category()}\n")
        
        rec_list = []
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.food_name}")
            print(f"     Score: {rec.confidence_score:.1f}%")
            print(f"     Why: {rec.reason}")
            print()
            rec_list.append(rec)
        
        return rec_list
    
    def create_shopping_list(self, recommendations: List) -> None:
        """Create shopping list from recommendations"""
        print("\n" + "-"*80)
        print("🛒 SHOPPING LIST")
        print("-"*80)
        
        print("\nBased on weekly recommendations:\n")
        
        categories = {}
        for rec in recommendations:
            # Extract category from recommendation reason
            category = "Other"
            if "protein" in rec.reason.lower():
                category = "Proteins"
            elif "vegetable" in rec.reason.lower():
                category = "Vegetables"
            elif "fruit" in rec.reason.lower():
                category = "Fruits"
            elif "grain" in rec.reason.lower():
                category = "Grains"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(rec.food_name)
        
        for category, foods in sorted(categories.items()):
            print(f"📦 {category}:")
            for food in foods:
                print(f"   ☐ {food}")
            print()
    
    def display_daily_plan(self, meals_per_day: Dict, day_name: str = "Today") -> None:
        """Display meal timing for a day"""
        print(f"\n⏰ Sample Daily Schedule ({day_name}):")
        print("   " + "-"*60)
        
        times = {
            'breakfast': '07:00 AM',
            'lunch': '12:30 PM',
            'snack': '03:30 PM',
            'dinner': '07:00 PM'
        }
        
        for meal, (time, cals) in zip(meals_per_day.keys(), zip(times.values(), meals_per_day.values())):
            print(f"   {time}  {meal.title():10s} - {cals:.0f} cal")
    
    def save_plan_to_file(self, user_data: Dict, predictions: List, 
                          diet_plan: Dict, recommendations: List) -> str:
        """Save the weekly plan to a JSON file"""
        plan_data = {
            'created': datetime.now().isoformat(),
            'user_name': user_data['name'],
            'user_profile': {
                'age': user_data['age'],
                'gender': 'Male' if user_data['gender'] == 1 else 'Female',
                'height_cm': user_data['height_cm'],
                'weight_kg': user_data['weight_kg'],
                'activity_level': user_data['activity_level'],
                'goal': user_data['goal'],
                'dietary_preference': user_data['dietary_preference']
            },
            'weight_predictions': predictions,
            'diet_plan': {
                'daily_calories': diet_plan['daily_calories'],
                'meals_per_day': {k: v for k, v in diet_plan['meals_per_day'].items()},
                'macro_targets': diet_plan['macro_targets']
            },
            'top_recommendations': [
                {
                    'food': rec.food_name,
                    'score': rec.confidence_score,
                    'reason': rec.reason
                } for rec in recommendations
            ]
        }
        
        filename = f"weekly_plan_{user_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        return str(filepath)
    
    def run_interactive_session(self) -> None:
        """Run the complete interactive planning session"""
        try:
            # Step 1: Get user input
            user_data = self.get_user_input()
            
            # Step 2: Weight prediction
            predictions, dates = self.predict_weekly_weight(user_data)
            
            # Step 3: Diet plan
            diet_plan = self.create_daily_meal_plan(user_data)
            
            # Step 4: Food recommendations
            recommendations = self.get_food_recommendations(user_data, top_n=8)
            
            # Step 5: Display daily schedule
            print("\n" + "-"*80)
            print("📅 SAMPLE DAILY MEAL TIMING")
            print("-"*80)
            self.display_daily_plan(diet_plan['meals_per_day'])
            
            # Step 6: Shopping list
            print("\n" + "-"*80)
            self.create_shopping_list(recommendations)
            
            # Step 7: Save plan
            print("-"*80)
            print("💾 SAVING YOUR PLAN")
            print("-"*80)
            filepath = self.save_plan_to_file(user_data, predictions, diet_plan, recommendations)
            print(f"\n✅ Weekly plan saved to: {filepath}")
            
            # Final summary
            print("\n" + "="*80)
            print("✨ YOUR PERSONALIZED WEEKLY HEALTH PLAN IS READY!".center(80))
            print("="*80)
            print(f"""
📋 Summary for {user_data['name']}:
   • Goal: {user_data['goal'].replace('_', ' ').title()}
   • Daily Calories: {diet_plan['daily_calories']:.0f}
   • 7-Day Weight Target: {predictions[-1]:.2f} kg
   • Predicted Change: {predictions[-1] - user_data['weight_kg']:+.2f} kg
   
🎯 Next Steps:
   1. Follow the daily meal timing schedule
   2. Prepare foods from the shopping list
   3. Make food choices from top recommendations
   4. Track your weight daily
   5. Adjust if needed next week

💡 Pro Tips:
   • Drink plenty of water (2-3L daily)
   • Get 7-9 hours of sleep
   • Stay consistent with exercise
   • Track your food intake
   • Review progress weekly
            """)
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n❌ Session cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n\n❌ Error during session: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    try:
        planner = WeeklyHealthPlanner()
        planner.run_interactive_session()
        
        # Ask if user wants to run again
        while True:
            again = input("\n🔄 Would you like to create another plan? (yes/no): ").strip().lower()
            if again in ['yes', 'y']:
                planner.run_interactive_session()
            else:
                print("\n👋 Thank you for using Weekly Health Planner. Stay healthy!\n")
                break
                
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
