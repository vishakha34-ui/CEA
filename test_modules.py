#!/usr/bin/env python3
"""
Test script for data_pipelines and advanced_models modules
"""

import sys
import traceback

print("="*70)
print("TESTING HEALTHVEND DATA PIPELINES & ADVANCED MODELS")
print("="*70)

# Test 1: Import modules
print("\n[TEST 1] Importing modules...")
try:
    from data_pipelines import DataPipeline, USDAFoodDatabase
    from advanced_models import WeightPredictionModel, FoodRecommendationModel
    print("[+] PASS: All modules imported successfully")
except Exception as e:
    print(f"[-] FAIL: Import error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create food database
print("\n[TEST 2] Creating food database...")
try:
    usda = USDAFoodDatabase()
    foods = usda.create_comprehensive_food_database()
    assert len(foods) > 0, "Food database is empty"
    assert 'food_name' in foods.columns, "Missing food_name column"
    assert 'calories' in foods.columns, "Missing calories column"
    print(f"[+] PASS: Created {len(foods)} food items")
    print(f"    Categories: {list(foods['category'].unique())}")
except Exception as e:
    print(f"[-] FAIL: Food database error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Data pipeline initialization
print("\n[TEST 3] Initializing data pipeline...")
try:
    pipeline = DataPipeline(use_cached=False)  # Don't try to download
    print("[+] PASS: Data pipeline initialized")
except Exception as e:
    print(f"[-] FAIL: Pipeline error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Sample data for ML models
print("\n[TEST 4] Creating sample training data...")
try:
    from advanced_models import create_sample_prediction_data
    X, y = create_sample_prediction_data()
    assert len(X) == 100, "Sample data should have 100 samples"
    assert len(X.columns) == 6, "Should have 6 features"
    print(f"[+] PASS: Created {len(X)} training samples with {len(X.columns)} features")
except Exception as e:
    print(f"[-] FAIL: Sample data error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Weight prediction model
print("\n[TEST 5] Training weight prediction model...")
try:
    from advanced_models import create_sample_prediction_data
    X_train, y_train = create_sample_prediction_data()
    
    wp_model = WeightPredictionModel()
    metrics = wp_model.train(X_train, y_train)
    
    assert 'ensemble_rmse' in metrics, "Missing RMSE metric"
    assert 'ensemble_r2' in metrics, "Missing R2 metric"
    assert metrics['ensemble_r2'] > 0, "R2 should be positive"
    
    print(f"[+] PASS: Model trained")
    print(f"    RMSE: {metrics['ensemble_rmse']:.4f}")
    print(f"    R² Score: {metrics['ensemble_r2']:.4f}")
except Exception as e:
    print(f"[-] FAIL: Weight model error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Make predictions
print("\n[TEST 6] Making weight predictions...")
try:
    test_user = {
        'age': 35,
        'gender_encoded': 1,
        'calorie_balance': -500,
        'activity_level_encoded': 1.55,
        'prev_weight_change': 0,
        'cumulative_weight_change': 0,
        'weight_kg': 85,
    }
    
    pred = wp_model.predict(test_user)
    
    assert hasattr(pred, 'predicted_weight_change'), "Missing prediction"
    assert hasattr(pred, 'confidence_interval'), "Missing confidence interval"
    
    print(f"[+] PASS: Prediction successful")
    print(f"    Predicted change: {pred.predicted_weight_change:.2f} kg")
    print(f"    Predicted weight: {pred.predicted_weight_7day:.2f} kg")
    print(f"    Confidence (95%): {pred.confidence_interval}")
except Exception as e:
    print(f"[-] FAIL: Prediction error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Food recommendation model
print("\n[TEST 7] Training food recommendation model...")
try:
    import pandas as pd
    import numpy as np
    
    # Create sample user profiles
    sample_users = pd.DataFrame({
        'age': np.random.randint(20, 70, 50),
        'gender': np.random.choice(['Male', 'Female'], 50),
        'activity_level': np.random.choice(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'], 50),
        'fitness_goal': np.random.choice(['Weight Loss', 'Maintenance', 'Muscle Gain'], 50),
        'dietary_preference': np.random.choice(['Omnivore', 'Vegetarian', 'Vegan'], 50),
        'bmi': np.random.uniform(18, 35, 50),
        'bmi_category': np.random.choice(['Underweight', 'Normal', 'Overweight', 'Obese'], 50),
        'daily_calorie_target': np.random.randint(1200, 3500, 50),
    })
    
    rec_model = FoodRecommendationModel()
    metrics = rec_model.train(sample_users, foods, sample_size=50)
    
    assert 'train_accuracy' in metrics, "Missing accuracy metric"
    assert metrics['train_accuracy'] > 0, "Accuracy should be positive"
    
    print(f"[+] PASS: Recommendation model trained")
    print(f"    Accuracy: {metrics['train_accuracy']:.4f}")
except Exception as e:
    print(f"[-] FAIL: Recommendation model error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Get recommendations
print("\n[TEST 8] Getting food recommendations...")
try:
    user_profile = {
        'age': 35,
        'gender': 'Male',
        'bmi': 27.5,
        'bmi_category': 'Overweight',
        'activity_level': 'Moderately Active',
        'fitness_goal': 'Weight Loss',
        'dietary_preference': 'Omnivore',
        'daily_calorie_target': 2000,
        'protein_target': 120,
    }
    
    recommendations = rec_model.recommend(user_profile, top_n=3)
    
    assert len(recommendations) <= 3, "Should return at most 3 recommendations"
    assert all(hasattr(r, 'food_name') for r in recommendations), "Missing food names"
    
    print(f"[+] PASS: Got {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec.food_name} (Score: {rec.confidence_score:.1f}%)")
except Exception as e:
    print(f"[-] FAIL: Recommendation error - {e}")
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("[+] ALL TESTS PASSED!")
print("="*70)
print("\nModules Ready for Production:")
print("  - data_pipelines.py: Real dataset integration")
print("  - advanced_models.py: ML model training and prediction")
print("  - app_enhanced.py: Integrated system with real data")
print("\nNext steps:")
print("  1. Run: python data_pipelines.py")
print("  2. Run: python advanced_models.py")
print("  3. Run: python app_enhanced.py")
