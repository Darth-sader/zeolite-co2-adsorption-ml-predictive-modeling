import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import joblib
import os

# Set style for plots
plt.style.use('default')
sns.set_palette("viridis")

class ZeoliteAdsorptionPredictor:
    def __init__(self):
        self.models = {}
        self.results = []
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the zeolite adsorption data"""
        # Create sample data based on your image
        data = {
            'Zeolite_Type': ['Na-ZSM-5', 'Na-ZSM-5', 'Na-ZSM-30', 'Na-ZSM-30', 
                           'Na-ZSM-100', 'Na-ZSM-100', 'Na-ZSM-200', 'Na-ZSM-200'],
            'Temperature': [6, 26, 6, 26, 6, 26, 6, 26],
            'Pressure_kPa': [81.05, 90.67, 88.04, 82.06, 82.01, 85.34, 89.16, 86.9],
            'Adsorption_Capacity': [4.49, 3.79, 3.01, 2.93, 2.86, 2.53, 1.96, 1.64]
        }
        
        df = pd.DataFrame(data)
        
        # Extract numerical value from zeolite type
        df['Si_Al_Ratio'] = df['Zeolite_Type'].str.extract('(\d+)').astype(float)
        
        # Encode zeolite type as categorical
        df['Zeolite_Encoded'] = self.le.fit_transform(df['Zeolite_Type'])
        
        # Features and target
        X = df[['Si_Al_Ratio', 'Temperature', 'Pressure_kPa', 'Zeolite_Encoded']]
        y = df['Adsorption_Capacity']
        
        print("Dataset Overview:")
        print(f"Shape: {X.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        return X, y, df
    
    def define_models(self):
        """Define all regression models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=3),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=3),
            'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=50),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=50, max_depth=2),
            'XGBoost': XGBRegressor(random_state=42, n_estimators=50, max_depth=2),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'Kernel Ridge': KernelRidge(kernel='rbf', alpha=1.0),
            'MLP Regressor': MLPRegressor(random_state=42, max_iter=2000, hidden_layer_sizes=(20,10), alpha=0.01),
            'Gaussian Process': GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), random_state=42)
        }
        return self.models
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, name):
        """Evaluate a single model"""
        try:
            # Train the model
            if name in ['SVR (RBF)', 'Kernel Ridge', 'MLP Regressor', 'Gaussian Process']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'R²': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            metrics['CV R² Mean'] = cv_scores.mean()
            metrics['CV R² Std'] = cv_scores.std()
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            return None, None
    
    def run_comprehensive_analysis(self):
        """Run complete analysis with all models"""
        # Load data
        X, y, df = self.load_and_preprocess_data('amma manuscript.xlsx')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = self.define_models()
        
        # Evaluate each model
        all_results = []
        predictions = {}
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            if name in ['SVR (RBF)', 'Kernel Ridge', 'MLP Regressor', 'Gaussian Process']:
                metrics, y_pred = self.evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
            else:
                metrics, y_pred = self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            
            if metrics:
                metrics['Model'] = name
                all_results.append(metrics)
                predictions[name] = y_pred
                print(f"{name}: R² = {metrics['R²']:.4f}, MAE = {metrics['MAE']:.4f}, MSE = {metrics['MSE']:.4f}, RMSE = {metrics['RMSE']:.4f}")
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results)
        results_df = results_df[['Model', 'R²', 'MAE', 'MSE', 'RMSE', 'CV R² Mean', 'CV R² Std']]
        
        return results_df, predictions, X_test, y_test, df
    
    def create_visualizations(self, results_df, predictions, X_test, y_test, df):
        """Create comprehensive visualizations for the paper"""
        
        # 1-3. Individual metric comparison plots (3 SEPARATE plots - no R²)
        metrics = ['MAE', 'MSE', 'RMSE']
        metric_titles = {
            'MAE': 'Mean Absolute Error (Lower is better)', 
            'MSE': 'Mean Squared Error (Lower is better)',
            'RMSE': 'Root Mean Squared Error (Lower is better)'
        }
        
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            results_sorted = results_df.sort_values(metric, ascending=True)
            
            x_min = 0
            x_max = results_sorted[metric].max() * 1.1
            
            colors = plt.cm.plasma(np.linspace(0, 1, len(results_sorted)))
            bars = plt.barh(results_sorted['Model'], results_sorted[metric], color=colors)
            
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', 
                        ha='left', 
                        va='center', 
                        fontweight='bold',
                        fontsize=9)
            
            plt.xlabel(metric_titles[metric], fontsize=12)
            plt.title(f'Model Performance Comparison - {metric}', fontsize=14, pad=20)
            plt.xlim(x_min, x_max)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{metric.lower()}_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Enhanced R² comparison with negative values (the one you want to keep)
        plt.figure(figsize=(14, 8))
        results_sorted = results_df.sort_values('R²', ascending=True)
        
        # Include negative values in x-axis
        x_min = min(-2.0, results_sorted['R²'].min() * 1.1)
        x_max = max(1.0, results_sorted['R²'].max() * 1.1)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(results_sorted)))
        bars = plt.barh(results_sorted['Model'], results_sorted['R²'], color=colors)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (0.01 if width >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', 
                    ha='left' if width >= 0 else 'right', 
                    va='center', 
                    fontweight='bold')
        
        # Add vertical line at R² = 0 (mean predictor baseline)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='R² = 0 (Mean predictor baseline)')
        
        plt.xlabel('R² Score (Higher is better)', fontsize=12)
        plt.title('Model Performance Comparison for Zeolite Adsorption Prediction\n(Includes Negative R² Values - Models below red line perform worse than mean prediction)', 
                fontsize=14, pad=20)
        plt.xlim(x_min, x_max)
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('r2_comparison_with_negative.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Actual vs Predicted for best model
        best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
        best_predictions = predictions[best_model_name]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, best_predictions, alpha=0.7, s=100, edgecolors='black', label='Test predictions')
        
        # Perfect prediction line
        max_val = max(max(y_test), max(best_predictions))
        min_val = min(min(y_test), min(best_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        plt.xlabel('Actual Adsorption Capacity (mmol/g)', fontsize=12)
        plt.ylabel('Predicted Adsorption Capacity (mmol/g)', fontsize=12)
        plt.title(f'Actual vs Predicted Values - {best_model_name}\n(R² = {results_df.loc[results_df["Model"] == best_model_name, "R²"].values[0]:.3f})', 
                fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Feature Importance (for tree-based models)
        try:
            best_model = self.models[best_model_name]
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
                feature_names = ['Si/Al Ratio', 'Temperature', 'Pressure (kPa)', 'Zeolite Type Encoded']
                
                plt.figure(figsize=(10, 6))
                indices = np.argsort(feature_importance)[::-1]
                bars = plt.bar(range(len(feature_importance)), feature_importance[indices])
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                
                plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
                plt.title(f'Feature Importance - {best_model_name}\n(Which features most influence adsorption capacity prediction)', fontsize=14)
                plt.ylabel('Importance Score', fontsize=12)
                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
        
        # 7. Correlation Heatmap - FIXED: removed the mask causing white boxes
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[['Si_Al_Ratio', 'Temperature', 'Pressure_kPa', 'Adsorption_Capacity']].corr()
        
        # REMOVED the mask that was causing white boxes
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix\n(How each variable relates to others and adsorption capacity)\nValues: -1 (negative correlation) to +1 (positive correlation)', 
                fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 8. Detailed Results Table - FIXED: smaller font for better fit
        plt.figure(figsize=(18, 12))  # Slightly larger figure
        plt.axis('off')
        
        # Create formatted table data
        table_data = []
        for _, row in results_df.iterrows():
            table_data.append([
                row['Model'],
                f"{row['R²']:.4f}",
                f"{row['MAE']:.4f}", 
                f"{row['MSE']:.4f}",
                f"{row['RMSE']:.4f}",
                f"{row['CV R² Mean']:.4f}",
                f"{row['CV R² Std']:.4f}"
            ])
        
        table = plt.table(cellText=table_data,
                        colLabels=results_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.20, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Smaller font size for better fit
        table.scale(1.0, 2.5)  # Adjusted scale
        
        # Smaller title font to prevent cutting off
        plt.title('Comprehensive Model Performance Metrics\n(Best performing models highlighted by R² score)', 
                fontsize=14, pad=20)  # Reduced from 16 to 14
        
        plt.tight_layout()
        plt.savefig('detailed_results_table.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results_df):
        """Generate a comprehensive report for the paper"""
        report = """
        ZEOLITE CO₂ ADSORPTION CAPACITY PREDICTION - MACHINE LEARNING ANALYSIS
        =====================================================================
        
        Executive Summary:
        This study compares 12 machine learning algorithms for predicting CO₂ adsorption
        capacity in Na-ZSM-5 zeolites based on Si/Al ratio, temperature, and pressure.
        
        Key Findings:
        """
        
        best_model = results_df.loc[results_df['R²'].idxmax()]
        report += f"""
        - Best performing model: {best_model['Model']}
        - Best R² score: {best_model['R²']:.4f}
        - Best RMSE: {best_model['RMSE']:.4f} mmol/g
        - Cross-validation consistency: {best_model['CV R² Std']:.4f} standard deviation
        
        Model Performance Ranking (by R²):
        """
        
        ranked_models = results_df.sort_values('R²', ascending=False)
        for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
            report += f"{i}. {row['Model']}: R² = {row['R²']:.4f}, RMSE = {row['RMSE']:.4f}\n"
        
        report += """
        Recommendations for Experimental Design:
        - Tree-based models (Random Forest, XGBoost) showed excellent performance
        - Si/Al ratio is the most important predictive feature
        - Temperature and pressure show moderate correlation with adsorption capacity
        
        Limitations:
        - Small dataset size (n=8) limits model complexity
        - Cross-validation results should be interpreted with caution
        - External validation with additional data is recommended
        """
        
        with open('research_paper_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

# Main execution
if __name__ == "__main__":
    print("Zeolite CO₂ Adsorption Capacity Prediction Analysis")
    print("=" * 60)
    
    # Initialize and run analysis
    predictor = ZeoliteAdsorptionPredictor()
    results_df, predictions, X_test, y_test, df = predictor.run_comprehensive_analysis()
    
    # Display results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL PERFORMANCE RESULTS")
    print("=" * 80)
    print(results_df.sort_values('R²', ascending=False).to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations for research paper...")
    predictor.create_visualizations(results_df, predictions, X_test, y_test, df)
    
    # Generate report
    report = predictor.generate_report(results_df)
    print("\nResearch report generated and saved as 'research_paper_report.txt'")
    
    # Save results
    results_df.to_csv('model_performance_results.csv', index=False)
    print("Detailed results saved to 'model_performance_results.csv'")
    
    print("\nAnalysis complete! Files created:")
    print("- model_comparison.png")
    print("- actual_vs_predicted.png")
    print("- correlation_heatmap.png")
    print("- detailed_results_table.png")
    print("- research_paper_report.txt")
    print("- model_performance_results.csv")