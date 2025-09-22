#!/usr/bin/env python3
"""
Detailed analysis of every sentence in the report to verify accuracy
"""

from docx import Document
import json
import re

def get_actual_ml_workflow():
    """Get actual ML workflow details from notebook"""
    return {
        'algorithms': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Extra Trees'],
        'preprocessing': {
            'missing_values': 'TotalCharges column',
            'categorical_encoding': 'OneHotEncoder with drop=first',
            'numerical_scaling': 'StandardScaler',
            'pipeline': 'ColumnTransformer'
        },
        'feature_engineering': ['TenureGroup', 'ChargesGroup'],
        'hyperparameter_tuning': 'GridSearchCV with 5-fold cross-validation',
        'feature_selection': 'SelectKBest with f_classif',
        'model_evaluation': ['accuracy_score', 'confusion_matrix', 'classification_report'],
        'model_saving': 'joblib.dump',
        'dataset_size': 7043,
        'churn_rate': 0.664
    }

def analyze_every_sentence():
    """Analyze every sentence in the report for accuracy"""
    try:
        doc = Document('individual_report.docx')
        ml_workflow = get_actual_ml_workflow()
        
        print("=== DETAILED SENTENCE-BY-SENTENCE ANALYSIS ===\n")
        
        issues = []
        correct_statements = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if not text or len(text) < 10:  # Skip empty or very short paragraphs
                continue
                
            print(f"--- Paragraph {i}: {text[:100]}...")
            
            # Check for specific technical claims
            sentence_issues = check_technical_accuracy(text, ml_workflow)
            if sentence_issues:
                issues.extend(sentence_issues)
                for issue in sentence_issues:
                    print(f"  ❌ ISSUE: {issue}")
            else:
                print(f"  ✅ OK")
            
            print()
        
        print("=== SUMMARY OF ISSUES ===")
        if issues:
            for issue in issues:
                print(f"❌ {issue}")
        else:
            print("✅ No technical accuracy issues found!")
        
        print(f"\nTotal issues found: {len(issues)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def check_technical_accuracy(text, ml_workflow):
    """Check if a text contains accurate technical information"""
    issues = []
    text_lower = text.lower()
    
    # Check algorithm mentions
    if 'algorithm' in text_lower or 'model' in text_lower:
        for algo in ml_workflow['algorithms']:
            if algo.lower() in text_lower:
                # Check if it's mentioned correctly
                if 'logistic regression' in text_lower and 'logistic' not in text_lower:
                    issues.append(f"Incorrect algorithm mention: {algo}")
    
    # Check preprocessing claims
    if 'preprocessing' in text_lower or 'preprocess' in text_lower:
        if 'labelencoder' in text_lower:
            issues.append("Mentions LabelEncoder (should be OneHotEncoder)")
        if 'standardscaler' in text_lower and 'columnTransformer' not in text_lower:
            issues.append("Mentions StandardScaler without ColumnTransformer context")
    
    # Check feature engineering
    if 'feature engineering' in text_lower or 'engineered' in text_lower:
        if 'tenuregroup' in text_lower and 'chargesgroup' in text_lower:
            pass  # Correct
        elif 'feature engineering' in text_lower:
            issues.append("Mentions feature engineering but may not list correct features")
    
    # Check hyperparameter tuning
    if 'hyperparameter' in text_lower or 'tuning' in text_lower:
        if 'gridsearchcv' not in text_lower and 'grid search' not in text_lower:
            issues.append("Mentions hyperparameter tuning but not GridSearchCV")
        if 'cross-validation' in text_lower and '5-fold' not in text_lower:
            issues.append("Mentions cross-validation but not 5-fold")
    
    # Check dataset claims
    if 'dataset' in text_lower or 'customers' in text_lower:
        if re.search(r'\b\d{4,5}\b', text):
            # Check if it mentions correct dataset size
            if '7043' not in text and '7,043' not in text:
                if 'customers' in text_lower:
                    issues.append("Mentions customer count but may not be 7,043")
    
    # Check churn rate claims
    if 'churn rate' in text_lower or '66' in text:
        if '66.4%' not in text and '0.664' not in text:
            issues.append("Mentions churn rate but may not be 66.4%")
    
    # Check accuracy claims
    if 'accuracy' in text_lower or '74' in text:
        if '74.2%' in text or '74.0%' in text:
            pass  # These are correct rounded values
        elif '74.17%' in text or '73.95%' in text:
            pass  # These are also correct
    
    # Check model saving
    if 'save' in text_lower and 'model' in text_lower:
        if 'joblib' not in text_lower and 'pickle' not in text_lower:
            issues.append("Mentions model saving but not joblib")
    
    # Check evaluation metrics
    if 'evaluation' in text_lower or 'metric' in text_lower:
        if 'confusion matrix' in text_lower or 'classification report' in text_lower:
            pass  # Correct metrics mentioned
    
    return issues

if __name__ == "__main__":
    analyze_every_sentence()
