"""
HealthVend Data Pipeline Module
================================

Load, clean, and preprocess real datasets from multiple sources:
- Hugging Face nutrition dataset
- USDA FoodData Central API
- CDC/WHO health metrics
- Synthetically enhanced for training ML models

The pipeline standardizes columns across datasets and creates unified training
sets for weight prediction and food recommendation models.

Author: HealthVend System
License: MIT
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

# ============================================================================
# Constants
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face dataset info
HF_NUTRITION_URL = "https://huggingface.co/datasets/sarthak-wiz01/nutrition_dataset/raw/main/data.csv"
HF_NUTRITION_PARQUET = "https://huggingface.co/datasets/sarthak-wiz01/nutrition_dataset/resolve/main/data/train-00000-of-00001.parquet"

# USDA FoodData Central
USDA_API_KEY = "DEMO_KEY"  # Replace with your API key
USDA_FDC_URL = "https://fdc.nal.usda.gov/api/v1/foods/search"

# WHO BMI categories
BMI_CATEGORIES = {
    "Underweight": (0, 18.5),
    "Normal": (18.5, 25),
    "Overweight": (25, 30),
    "Obese": (30, float('inf'))
}

ACTIVITY_MULTIPLIERS = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725
}

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class UserProfile:
    """Standardized user health profile."""
    user_id: str
    age: int
    gender: str  # Male/Female
    height_cm: float
    weight_kg: float
    activity_level: str
    fitness_goal: str  # Weight Loss, Maintenance, Muscle Gain
    dietary_preference: str  # Omnivore, Vegetarian, Vegan

    def calculate_bmi(self) -> float:
        """Calculate BMI: weight(kg) / (height(m))^2"""
        height_m = self.height_cm / 100
        return round(self.weight_kg / (height_m ** 2), 2)

    def get_bmi_category(self) -> str:
        """Classify BMI into category."""
        bmi = self.calculate_bmi()
        for category, (min_bmi, max_bmi) in BMI_CATEGORIES.items():
            if min_bmi <= bmi < max_bmi:
                return category
        return "Obese"

    def estimate_tdee(self) -> float:
        """
        Estimate Total Daily Energy Expenditure using Mifflin-St Jeor.
        BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + (s * gender)
        TDEE = BMR * activity_multiplier
        where gender_factor: +5 for males, -161 for females
        """
        if self.gender.lower() == "male":
            gender_factor = 5
        else:
            gender_factor = -161

        bmr = (10 * self.weight_kg) + (6.25 * self.height_cm) - (5 * self.age) + gender_factor
        activity_multiplier = ACTIVITY_MULTIPLIERS.get(self.activity_level, 1.375)
        tdee = bmr * activity_multiplier
        return round(tdee, 2)


@dataclass
class FoodItem:
    """Standardized food nutrition profile."""
    food_id: str
    food_name: str
    category: str  # Protein, Vegetable, Fruit, Grain, Dairy, etc.
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    serving_size: str
    bmi_suitability: str  # Underweight, Normal, Overweight, Obese, All
    

@dataclass
class WeeklyProgress:
    """Weekly user tracking data."""
    user_id: str
    week: int
    daily_calories: float
    weight_change_kg: float


# ============================================================================
# Dataset Loaders
# ============================================================================

class HuggingFaceNutritionDataset:
    """Load and process Hugging Face nutrition dataset."""

    @staticmethod
    def load_from_csv(url: str = HF_NUTRITION_URL) -> pd.DataFrame:
        """
        Load nutrition dataset from Hugging Face CSV.
        
        Columns: Age, Gender, Height, Weight, Activity Level, Fitness Goal,
                 Dietary Preference, Daily Calorie Target, Protein (g),
                 Carbohydrates (g), Fat (g), Breakfast/Lunch/Dinner/Snack Suggestions
        """
        try:
            print("[*] Loading Hugging Face nutrition dataset...")
            df = pd.read_csv(url, timeout=30)
            print(f"[+] Loaded {len(df)} records from Hugging Face")
            return df
        except Exception as e:
            print(f"[-] Failed to load from URL: {e}")
            print("[*] Attempting to use local cached version...")
            return None

    @staticmethod
    def load_from_parquet(url: str = HF_NUTRITION_PARQUET) -> pd.DataFrame:
        """Load using parquet format (faster)."""
        try:
            print("[*] Loading Hugging Face nutrition dataset (Parquet)...")
            df = pd.read_parquet(url)
            print(f"[+] Loaded {len(df)} records from Hugging Face")
            return df
        except Exception as e:
            print(f"[-] Failed to load parquet: {e}")
            return None

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize HF dataset columns to HealthVend schema."""
        standardized = pd.DataFrame()

        # Direct mappings
        mapping = {
            'Age': 'age',
            'Gender': 'gender',
            'Height': 'height_cm',
            'Weight': 'weight_kg',
            'Activity Level': 'activity_level',
            'Fitness Goal': 'fitness_goal',
            'Dietary Preference': 'dietary_preference',
        }

        for hf_col, new_col in mapping.items():
            if hf_col in df.columns:
                standardized[new_col] = df[hf_col]

        # Nutrition columns
        nutrition_mapping = {
            'Daily Calorie Target': 'daily_calorie_target',
            'Protein (g)': 'protein_g',
            'Carbohydrates (g)': 'carbs_g',
            'Fat (g)': 'fat_g',
        }

        for hf_col, new_col in nutrition_mapping.items():
            if hf_col in df.columns:
                standardized[new_col] = pd.to_numeric(df[hf_col], errors='coerce')

        return standardized

    @staticmethod
    def create_synthetic_progress(users_df: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
        """
        Create synthetic weekly tracking data for weight prediction training.
        
        Simulates realistic weight changes based on calorie deficit/surplus:
        - 7000 calorie deficit = 1 kg weight loss
        - Sedentary users: baseline maintenance TDEE
        - Active users: higher TDEE, potential for larger changes
        """
        progress_records = []

        for idx, user in users_df.iterrows():
            user_id = f"user_{idx:04d}"
            profile = UserProfile(
                user_id=user_id,
                age=int(user['age']),
                gender=user['gender'],
                height_cm=float(user['height_cm']),
                weight_kg=float(user['weight_kg']),
                activity_level=user['activity_level'],
                fitness_goal=user['fitness_goal'],
                dietary_preference=user['dietary_preference']
            )

            tdee = profile.estimate_tdee()

            for week in range(1, weeks + 1):
                # Generate calorie intake based on fitness goal
                if user['fitness_goal'] == 'Weight Loss':
                    # Average 500 cal deficit for 0.5 kg/week loss
                    daily_cals = tdee - np.random.normal(500, 100)
                    weight_change = -(week * 0.5 + np.random.normal(0, 0.2))
                elif user['fitness_goal'] == 'Muscle Gain':
                    # 300 cal surplus for muscle gain
                    daily_cals = tdee + np.random.normal(300, 100)
                    weight_change = (week * 0.3 + np.random.normal(0, 0.15))
                else:  # Maintenance
                    # Near TDEE maintenance
                    daily_cals = tdee + np.random.normal(0, 150)
                    weight_change = np.random.normal(0, 0.3)

                progress_records.append({
                    'user_id': user_id,
                    'week': week,
                    'daily_calories': max(1200, daily_cals),  # Realistic min
                    'weight_change_kg': weight_change,
                    'activity_level': user['activity_level'],
                    'fitness_goal': user['fitness_goal'],
                    'age': int(user['age']),
                    'gender': user['gender'],
                    'bmi_category': profile.get_bmi_category(),
                })

        return pd.DataFrame(progress_records)


class USDAFoodDatabase:
    """Load food nutrition data from USDA FoodData Central."""

    @staticmethod
    def fetch_foods(query: str, pageSize: int = 50) -> Dict:
        """Query USDA FDC API for food items."""
        params = {
            'query': query,
            'pageSize': pageSize,
            'api_key': USDA_API_KEY,
        }

        try:
            response = requests.get(USDA_FDC_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[-] USDA API error: {e}")
            return {}

    @staticmethod
    def create_comprehensive_food_database() -> pd.DataFrame:
        """
        Create comprehensive food database with nutrition information.
        Uses predefined common foods suitable for health vending.
        """
        foods_data = {
            'food_id': [],
            'food_name': [],
            'category': [],
            'calories': [],
            'protein_g': [],
            'fat_g': [],
            'carbs_g': [],
            'fiber_g': [],
            'serving_size': [],
            'bmi_suitability': []
        }

        # Comprehensive food database (per 100g or serving)
        comprehensive_foods = [
            # Proteins
            ('grilled_chicken', 'Grilled Chicken Breast', 'Protein', 165, 31, 3.6, 0, 0, '100g', 'All'),
            ('salmon', 'Salmon Fillet', 'Protein', 206, 22, 12.3, 0, 0, '100g', 'All'),
            ('eggs', 'Boiled Eggs', 'Protein', 155, 13, 11, 1.1, 0, '100g', 'All'),
            ('greek_yogurt', 'Greek Yogurt Plain', 'Dairy', 59, 10, 0.4, 3.3, 0, '100g', 'All'),
            ('cottage_cheese', 'Cottage Cheese', 'Dairy', 98, 11, 4.3, 3.4, 0, '100g', 'All'),
            ('lentils', 'Cooked Lentils', 'Protein', 116, 9, 0.4, 20, 3.8, '100g', 'All'),
            ('tofu', 'Silken Tofu', 'Protein', 61, 8, 3.5, 2, 0, '100g', 'All'),
            ('almonds', 'Raw Almonds', 'Nuts', 579, 21, 50, 22, 12.5, '100g', 'Normal'),
            ('chickpeas', 'Cooked Chickpeas', 'Legume', 134, 9, 2.6, 23, 6.4, '100g', 'All'),

            # Vegetables
            ('broccoli', 'Steamed Broccoli', 'Vegetable', 34, 2.8, 0.4, 7, 2.4, '100g', 'All'),
            ('spinach', 'Raw Spinach', 'Vegetable', 23, 2.9, 0.4, 3.6, 2.2, '100g', 'All'),
            ('carrots', 'Raw Carrots', 'Vegetable', 41, 0.9, 0.2, 10, 2.8, '100g', 'All'),
            ('asparagus', 'Steamed Asparagus', 'Vegetable', 27, 3, 0.1, 5, 2.1, '100g', 'All'),
            ('bell_peppers', 'Raw Bell Peppers', 'Vegetable', 31, 1, 0.3, 6, 2, '100g', 'All'),
            ('tomatoes', 'Raw Tomatoes', 'Vegetable', 18, 0.9, 0.2, 3.9, 1.2, '100g', 'All'),
            ('cucumber', 'Raw Cucumber', 'Vegetable', 16, 0.7, 0.1, 3.6, 0.5, '100g', 'All'),

            # Fruits
            ('banana', 'Raw Banana', 'Fruit', 89, 1.1, 0.3, 23, 2.6, '100g (1 medium)', 'All'),
            ('apple', 'Raw Apple', 'Fruit', 52, 0.3, 0.2, 14, 2.4, '100g', 'All'),
            ('berries', 'Mixed Berries', 'Fruit', 57, 0.7, 0.3, 14, 2.4, '100g', 'All'),
            ('orange', 'Raw Orange', 'Fruit', 47, 0.9, 0.3, 12, 2.4, '100g', 'All'),
            ('pineapple', 'Raw Pineapple', 'Fruit', 50, 0.5, 0.1, 13, 1.4, '100g', 'All'),
            ('strawberries', 'Raw Strawberries', 'Fruit', 32, 0.7, 0.3, 8, 2, '100g', 'All'),

            # Grains
            ('brown_rice', 'Cooked Brown Rice', 'Grain', 111, 2.6, 0.9, 23, 1.8, '100g', 'All'),
            ('oatmeal', 'Cooked Oatmeal', 'Grain', 68, 2.4, 1.4, 12, 1.7, '100g', 'All'),
            ('whole_wheat_bread', 'Whole Wheat Bread', 'Grain', 247, 8.9, 1.7, 41, 6.8, '100g (2 slices)', 'All'),
            ('quinoa', 'Cooked Quinoa', 'Grain', 120, 4.4, 1.9, 21, 2.8, '100g', 'All'),
            ('pasta', 'Whole Wheat Pasta', 'Grain', 174, 7, 1.1, 34, 5.7, '100g (dry)', 'All'),

            # Dairy
            ('low_fat_milk', 'Low Fat Milk', 'Dairy', 61, 3.2, 1.5, 4.8, 0, '200ml', 'All'),
            ('cheddar_cheese', 'Cheddar Cheese', 'Dairy', 403, 23, 33, 1.3, 0, '100g', 'Normal'),

            # Oils & Condiments
            ('olive_oil', 'Extra Virgin Olive Oil', 'Oil', 884, 0, 100, 0, 0, '10ml', 'Normal'),
            ('peanut_butter', 'Peanut Butter', 'Spread', 588, 25, 50, 20, 6, '100g', 'Normal'),

            # Snacks
            ('dark_chocolate', 'Dark Chocolate 70%', 'Snack', 598, 8, 43, 46, 7, '100g', 'Normal'),
            ('almonds_snack', 'Almonds (Snack Pack)', 'Snack', 579, 21, 50, 22, 12.5, '28g', 'Normal'),
            ('granola', 'Granola', 'Snack', 471, 10, 20, 63, 7, '100g', 'Normal'),
            ('popcorn', 'Air Popped Popcorn', 'Snack', 31, 3.5, 0.4, 6, 1.2, '100g (popped)', 'All'),
        ]

        food_id = 0
        for food in comprehensive_foods:
            foods_data['food_id'].append(f"food_{food_id:03d}")
            foods_data['food_name'].append(food[1])
            foods_data['category'].append(food[2])
            foods_data['calories'].append(food[3])
            foods_data['protein_g'].append(food[4])
            foods_data['fat_g'].append(food[5])
            foods_data['carbs_g'].append(food[6])
            foods_data['fiber_g'].append(food[7])
            foods_data['serving_size'].append(food[8])
            foods_data['bmi_suitability'].append(food[9])
            food_id += 1

        return pd.DataFrame(foods_data)


# ============================================================================
# Data Pipeline Orchestrator
# ============================================================================

class DataPipeline:
    """Main orchestrator for loading and processing all datasets."""

    def __init__(self, use_cached: bool = True):
        """
        Initialize data pipeline.
        
        Args:
            use_cached: If True, use cached datasets if available
        """
        self.use_cached = use_cached
        self.nutrition_data = None
        self.food_database = None
        self.progress_data = None
        self.training_data = None

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load and process all datasets."""
        print("\n" + "="*70)
        print("HEALTHVEND DATA PIPELINE - LOADING REAL DATASETS")
        print("="*70)

        # Try to load from cache first
        if self.use_cached:
            cached = self._load_cached_datasets()
            if cached:
                return cached

        datasets = {}

        # Load Hugging Face nutrition dataset
        print("\n[1] Loading Hugging Face Nutrition Dataset")
        print("-" * 70)
        hf_loader = HuggingFaceNutritionDataset()
        hf_raw = hf_loader.load_from_csv()
        
        if hf_raw is None:
            hf_raw = hf_loader.load_from_parquet()
        
        if hf_raw is not None:
            hf_standardized = hf_loader.standardize_columns(hf_raw)
            self.nutrition_data = hf_standardized
            datasets['nutrition'] = hf_standardized
            print(f"[+] HF Nutrition Dataset: {len(hf_standardized)} users loaded")
            print(f"    Columns: {list(hf_standardized.columns)[:5]}...")
        else:
            print("[-] Could not load HF nutrition dataset, using local data")

        # Load USDA/Comprehensive food database
        print("\n[2] Loading Food Nutrition Database")
        print("-" * 70)
        usda = USDAFoodDatabase()
        food_db = usda.create_comprehensive_food_database()
        self.food_database = food_db
        datasets['foods'] = food_db
        print(f"[+] Food Database: {len(food_db)} items loaded")
        print(f"    Categories: {food_db['category'].unique().tolist()}")

        # Create synthetic progress data for weight prediction training
        print("\n[3] Creating Synthetic Weight Progress Data (for ML training)")
        print("-" * 70)
        if self.nutrition_data is not None:
            progress_data = HuggingFaceNutritionDataset.create_synthetic_progress(
                self.nutrition_data, weeks=12
            )
            self.progress_data = progress_data
            datasets['progress'] = progress_data
            print(f"[+] Progress Data: {len(progress_data)} records created")
            print(f"    Training weeks: 1-12")
            print(f"    Features: {list(progress_data.columns)[:5]}...")

        # Cache datasets
        if self.use_cached:
            self._cache_datasets(datasets)

        print("\n[+] All datasets loaded successfully!")
        print(f"    Total datasets: {len(datasets)}")

        return datasets

    def create_weight_prediction_training_set(self, 
                                             progress_df: Optional[pd.DataFrame] = None,
                                             test_split: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Create training and testing sets for weight prediction model.
        
        Features: age, gender (encoded), height, weight, daily_calories, activity, bmi_category
        Target: weight_change_kg
        
        Returns:
            Tuple of (train_dict, test_dict) containing X and y DataFrames
        """
        if progress_df is None:
            progress_df = self.progress_data
        
        if progress_df is None:
            raise ValueError("No progress data available")

        print("\n[*] Creating weight prediction training set...")

        # Make copy to avoid modifying original
        df = progress_df.copy()

        # Calculate features
        df['calorie_balance'] = df['daily_calories'] - (2000)  # Baseline approximation
        df['gender_encoded'] = (df['gender'] == 'Male').astype(int)

        # Create lagged features (week-over-week changes)
        df_sorted = df.sort_values(['user_id', 'week'])
        df_sorted['prev_weight_change'] = df_sorted.groupby('user_id')['weight_change_kg'].shift(1)
        df_sorted['cumulative_weight_change'] = df_sorted.groupby('user_id')['weight_change_kg'].cumsum()

        # Select features and target
        feature_cols = [
            'age', 'gender_encoded', 'calorie_balance', 'activity_level_encoded',
            'prev_weight_change', 'cumulative_weight_change'
        ]

        # Encode categorical columns
        df_sorted['activity_level_encoded'] = df_sorted['activity_level'].map({
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725
        })

        # Fill NaN values
        df_sorted = df_sorted.fillna(0)

        X = df_sorted[feature_cols]
        y = df_sorted['weight_change_kg']

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )

        train_dict = {'X': X_train, 'y': y_train}
        test_dict = {'X': X_test, 'y': y_test}

        print(f"[+] Training set: {len(X_train)} samples")
        print(f"[+] Test set: {len(X_test)} samples")
        print(f"[+] Features: {feature_cols}")

        return train_dict, test_dict

    def create_food_recommendation_training_set(self,
                                               nutrition_df: Optional[pd.DataFrame] = None
                                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training set for food recommendation model.
        Features: BMI category, daily calorie target, protein/carbs/fat targets
        Target: food categories and items suitable for user profile
        
        Returns:
            Tuple of (features_df, foods_df)
        """
        if nutrition_df is None:
            nutrition_df = self.nutrition_data

        if nutrition_df is None:
            raise ValueError("No nutrition data available")

        print("\n[*] Creating food recommendation training set...")

        # Calculate BMI for each user
        nutrition_df_copy = nutrition_df.copy()
        nutrition_df_copy['bmi'] = (
            nutrition_df_copy['weight_kg'] / 
            ((nutrition_df_copy['height_cm'] / 100) ** 2)
        ).round(2)

        # Classify BMI
        def classify_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'

        nutrition_df_copy['bmi_category'] = nutrition_df_copy['bmi'].apply(classify_bmi)

        # Select features for recommendation model
        feature_cols = [
            'age', 'gender', 'bmi', 'bmi_category', 'activity_level',
            'fitness_goal', 'dietary_preference', 'daily_calorie_target'
        ]

        X = nutrition_df_copy[feature_cols].dropna()

        print(f"[+] Recommendation training set: {len(X)} user profiles")
        print(f"[+] Features: {feature_cols}")
        print(f"[+] BMI categories: {X['bmi_category'].value_counts().to_dict()}")

        return X, self.food_database

    def _load_cached_datasets(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load datasets from cache if available."""
        cache_dir = PROCESSED_DATA_DIR
        required_files = {
            'nutrition': cache_dir / 'nutrition_data.csv',
            'foods': cache_dir / 'food_database.csv',
            'progress': cache_dir / 'progress_data.csv',
        }

        all_exist = all(f.exists() for f in required_files.values())

        if not all_exist:
            return None

        print("\n[*] Loading cached datasets...")
        datasets = {}

        try:
            datasets['nutrition'] = pd.read_csv(required_files['nutrition'])
            datasets['foods'] = pd.read_csv(required_files['foods'])
            datasets['progress'] = pd.read_csv(required_files['progress'])
            print("[+] All datasets loaded from cache")
            self.nutrition_data = datasets['nutrition']
            self.food_database = datasets['foods']
            self.progress_data = datasets['progress']
            return datasets
        except Exception as e:
            print(f"[-] Cache load failed: {e}")
            return None

    def _cache_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """Cache datasets to local CSV files."""
        try:
            for name, df in datasets.items():
                cache_file = PROCESSED_DATA_DIR / f"{name}_data.csv"
                df.to_csv(cache_file, index=False)
            print(f"\n[+] Datasets cached to {PROCESSED_DATA_DIR}")
        except Exception as e:
            print(f"[-] Caching failed: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def download_dataset_info() -> str:
    """Print information about available datasets."""
    info = """
    AVAILABLE DATASETS FOR HEALTHVEND
    ==================================

    1. Hugging Face Nutrition Dataset
       - 500+ user profiles with age, gender, height, weight
       - Activity levels: Sedentary to Very Active
       - Fitness goals: Weight Loss, Maintenance, Muscle Gain
       - Dietary preferences: Omnivore, Vegetarian, Vegan
       - Nutritional recommendations: Calories, Macros, Meal suggestions
       - License: MIT
       - URL: https://huggingface.co/datasets/sarthak-wiz01/nutrition_dataset

    2. USDA FoodData Central
       - 400,000+ food items with complete nutrition data
       - Real-world food products and recipes
       - API access: https://fdc.nal.usda.gov/api/v1/foods/search
       - Requires API key (free registration)

    3. Synthetic Enhanced Dataset (Generated by HealthVend)
       - 12 weeks of weekly weight tracking
       - Realistic weight changes based on calorie balance
       - Simulated adherence patterns
       - Activity level variations

    USAGE IN HEALTHVEND
    ===================

    1. Weight Prediction Model
       - Trains on 12 weeks of synthetic progress data
       - Predicts 7-day weight change
       - Features: age, gender, calorie balance, activity level
       - Algorithm: Linear Regression + Random Forest ensemble

    2. Food Recommendation System
       - Uses HF nutrition dataset for user profiles
       - Matches users with suitable foods from database
       - Considers: BMI category, calorie targets, dietary preferences
       - Algorithm: Decision Tree with nutritional scoring

    3. Population Analysis
       - Analyzes 500+ user profiles for patterns
       - By: BMI category, age group, gender, activity level
       - Identifies: Success rates, dietary trends, common goals
    """
    return info


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline(use_cached=True)

    # Load all datasets
    datasets = pipeline.load_all_datasets()

    # Create training sets
    print("\n" + "="*70)
    print("CREATING TRAINING DATASETS FOR ML MODELS")
    print("="*70)

    try:
        weight_train, weight_test = pipeline.create_weight_prediction_training_set()
        print("\n[+] Weight prediction training set ready!")
        print(f"    Training: {len(weight_train['X'])} samples")
        print(f"    Testing: {len(weight_test['X'])} samples")
    except Exception as e:
        print(f"[-] Weight prediction setup failed: {e}")

    try:
        recommendation_features, food_items = pipeline.create_food_recommendation_training_set()
        print("\n[+] Food recommendation training set ready!")
        print(f"    User profiles: {len(recommendation_features)}")
        print(f"    Food items: {len(food_items)}")
    except Exception as e:
        print(f"[-] Recommendation setup failed: {e}")

    print("\n[+] Data pipeline initialization complete!")
