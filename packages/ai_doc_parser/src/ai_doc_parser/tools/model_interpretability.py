"""
Model Interpretability Analysis Tool

This module provides comprehensive analysis tools for understanding why certain rows
are being classified the way they are by the RandomForestClassifier. It includes:

1. Feature Importance Analysis
2. SHAP (SHapley Additive exPlanations) Analysis
3. LIME (Local Interpretable Model-agnostic Explanations) Analysis

Author: AI Assistant
Date: 2024
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_text

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

log = logging.getLogger(__name__)


class ModelInterpretabilityAnalyzer:
    """
    A comprehensive tool for analyzing model interpretability using multiple methods.

    This class provides methods to understand why the RandomForestClassifier makes
    certain predictions through feature importance, SHAP, and LIME analysis.
    """

    def __init__(
        self, model: RandomForestClassifier, feature_names: List[str], class_names: Optional[List[str]] = None
    ):
        """
        Initialize the analyzer with a trained model.

        Args:
            model: Trained RandomForestClassifier
            feature_names: List of feature names used in the model
            class_names: Optional list of class names for better visualization
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or [f"Class_{i}" for i in range(len(model.classes_))]
        self.shap_explainer = None
        self.lime_explainer = None

        # Validate inputs
        if len(feature_names) != model.n_features_in_:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) doesn't match "
                f"model features ({model.n_features_in_})"
            )

    def analyze_feature_importance(
        self, top_n: int = 20, save_plot: bool = True, output_dir: str = "analysis_output"
    ) -> Dict[str, Any]:
        """
        Analyze and visualize feature importance from the RandomForest model.

        Args:
            top_n: Number of top features to display
            save_plot: Whether to save the plot to disk
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing feature importance data and plot
        """
        log.info("Analyzing feature importance...")

        # Get feature importance
        importance_scores = self.model.feature_importances_

        # Create feature importance dataframe
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importance_scores}).sort_values(
            'importance', ascending=False
        )

        # Get top N features
        top_features = importance_df.head(top_n)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Horizontal bar plot
        ax1.barh(range(len(top_features)), top_features['importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title(f'Top {top_n} Feature Importance')
        ax1.invert_yaxis()

        # Cumulative importance plot
        cumulative_importance = importance_df['importance'].cumsum()
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% of total importance')
        ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% of total importance')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'feature_importance_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"Feature importance plot saved to: {plot_path}")

        # Print summary statistics
        log.info(f"\nFeature Importance Analysis Summary:")
        log.info(f"Total features: {len(importance_df)}")
        log.info(f"Top {top_n} features account for {top_features['importance'].sum():.3f} of total importance")
        log.info(f"Features needed for 80% importance: {len(cumulative_importance[cumulative_importance <= 0.8])}")
        log.info(f"Features needed for 90% importance: {len(cumulative_importance[cumulative_importance <= 0.9])}")

        return {
            'importance_df': importance_df,
            'top_features': top_features,
            'cumulative_importance': cumulative_importance,
            'plot': fig,
        }

    def analyze_tree_structure(
        self, tree_index: int = 0, max_depth: int = 3, save_text: bool = True, output_dir: str = "analysis_output"
    ) -> str:
        """
        Analyze the structure of a specific tree in the RandomForest.

        Args:
            tree_index: Index of the tree to analyze
            max_depth: Maximum depth to display
            save_text: Whether to save the tree structure to file
            output_dir: Directory to save outputs

        Returns:
            String representation of the tree structure
        """
        log.info(f"Analyzing tree {tree_index} structure...")

        if tree_index >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_index} out of range. Model has {len(self.model.estimators_)} trees.")

        tree = self.model.estimators_[tree_index]

        # Export tree as text
        tree_text = export_text(tree, feature_names=self.feature_names, max_depth=max_depth)

        # Save to file if requested
        if save_text:
            os.makedirs(output_dir, exist_ok=True)
            tree_path = os.path.join(output_dir, f'tree_{tree_index}_structure.txt')
            with open(tree_path, 'w') as f:
                f.write(tree_text)
            log.info(f"Tree structure saved to: {tree_path}")

        log.info(f"Tree {tree_index} analysis completed")
        return tree_text

    def analyze_shap_values(
        self, X: pd.DataFrame, sample_size: int = 100, save_plots: bool = True, output_dir: str = "analysis_output"
    ) -> Dict[str, Any]:
        """
        Perform SHAP analysis for global and local explanations.

        Args:
            X: Feature matrix for analysis
            sample_size: Number of samples to use for SHAP analysis (for performance)
            save_plots: Whether to save plots to disk
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing SHAP values and explanations
        """
        if not SHAP_AVAILABLE:
            log.error("SHAP is not available. Install with: pip install shap")
            return {}

        log.info("Performing SHAP analysis...")

        # Sample data for performance if needed
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
            sample_indices = np.arange(len(X))

        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        shap_values = self.shap_explainer.shap_values(X_sample)

        # Handle multi-class case
        if isinstance(shap_values, list):
            # Multi-class: use the first class for global analysis
            shap_values_global = shap_values[0]
            class_idx = 0
        else:
            shap_values_global = shap_values
            class_idx = None

        # Create visualizations
        fig = plt.figure(figsize=(20, 12))

        # 1. Summary plot
        plt.subplot(2, 3, 1)
        shap.summary_plot(shap_values_global, X_sample, feature_names=self.feature_names, show=False, max_display=15)
        plt.title('SHAP Summary Plot')

        # 2. Feature importance from SHAP
        plt.subplot(2, 3, 2)
        shap_importance = np.abs(shap_values_global).mean(0)
        feature_importance_df = pd.DataFrame(
            {'feature': self.feature_names, 'shap_importance': shap_importance}
        ).sort_values('shap_importance', ascending=True)

        plt.barh(range(len(feature_importance_df)), feature_importance_df['shap_importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('SHAP Feature Importance')

        # 3. Waterfall plot for first sample
        plt.subplot(2, 3, 3)
        if class_idx is not None:
            shap.waterfall_plot(
                self.shap_explainer.expected_value[class_idx],
                shap_values[class_idx][0],
                X_sample.iloc[0],
                show=False,
                max_display=10,
            )
        else:
            shap.waterfall_plot(
                self.shap_explainer.expected_value, shap_values[0], X_sample.iloc[0], show=False, max_display=10
            )
        plt.title('SHAP Waterfall Plot (First Sample)')

        # 4. Partial dependence plot for top feature
        plt.subplot(2, 3, 4)
        top_feature_idx = np.argmax(shap_importance)
        top_feature_name = self.feature_names[top_feature_idx]
        shap.partial_dependence_plot(
            top_feature_idx,
            self.model.predict,
            X_sample,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            show=False,
        )
        plt.title(f'Partial Dependence: {top_feature_name}')

        # 5. SHAP values distribution
        plt.subplot(2, 3, 5)
        plt.hist(shap_values_global.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('SHAP Value')
        plt.ylabel('Frequency')
        plt.title('SHAP Values Distribution')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # 6. Feature correlation with SHAP values
        plt.subplot(2, 3, 6)
        correlations = []
        for i, feature in enumerate(self.feature_names):
            corr = np.corrcoef(X_sample.iloc[:, i], shap_values_global[:, i])[0, 1]
            correlations.append(corr)

        corr_df = pd.DataFrame({'feature': self.feature_names, 'correlation': correlations}).sort_values(
            'correlation', key=abs, ascending=True
        )

        plt.barh(range(len(corr_df)), corr_df['correlation'])
        plt.yticks(range(len(corr_df)), corr_df['feature'])
        plt.xlabel('Correlation with SHAP Values')
        plt.title('Feature-SHAP Correlation')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save plots if requested
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'shap_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"SHAP analysis plot saved to: {plot_path}")

        log.info("SHAP analysis completed")

        return {
            'shap_values': shap_values,
            'explainer': self.shap_explainer,
            'feature_importance': feature_importance_df,
            'correlations': corr_df,
            'plot': fig,
            'sample_indices': sample_indices,
        }

    def analyze_lime_explanations(
        self,
        X: pd.DataFrame,
        sample_indices: List[int] = None,
        num_features: int = 10,
        save_plots: bool = True,
        output_dir: str = "analysis_output",
    ) -> Dict[str, Any]:
        """
        Perform LIME analysis for local explanations.

        Args:
            X: Feature matrix for analysis
            sample_indices: Specific samples to explain (default: first 5)
            num_features: Number of features to show in explanations
            save_plots: Whether to save plots to disk
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing LIME explanations
        """
        if not LIME_AVAILABLE:
            log.error("LIME is not available. Install with: pip install lime")
            return {}

        log.info("Performing LIME analysis...")

        # Default to first 5 samples if none specified
        if sample_indices is None:
            sample_indices = list(range(min(5, len(X))))

        # Create LIME explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=True,
        )

        explanations = []
        predictions = []

        # Generate explanations for each sample
        for idx in sample_indices:
            if idx >= len(X):
                continue

            sample = X.iloc[idx]
            explanation = self.lime_explainer.explain_instance(
                sample.values, self.model.predict_proba, num_features=num_features
            )

            explanations.append(explanation)
            predictions.append(self.model.predict([sample.values])[0])

        # Create visualization
        n_samples = len(explanations)
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for i, (explanation, pred) in enumerate(zip(explanations, predictions)):
            # Get explanation data
            exp_list = explanation.as_list()

            # Create horizontal bar plot
            features = [item[0] for item in exp_list]
            weights = [item[1] for item in exp_list]

            colors = ['red' if w < 0 else 'green' for w in weights]
            axes[i].barh(range(len(features)), weights, color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('LIME Weight')
            axes[i].set_title(f'Sample {sample_indices[i]} - Predicted: {self.class_names[pred]}')
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()

        # Save plots if requested
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'lime_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"LIME analysis plot saved to: {plot_path}")

        log.info("LIME analysis completed")

        return {
            'explanations': explanations,
            'explainer': self.lime_explainer,
            'sample_indices': sample_indices,
            'predictions': predictions,
            'plot': fig,
        }

    def analyze_prediction_errors(
        self, X: pd.DataFrame, y_true: pd.Series, save_plots: bool = True, output_dir: str = "analysis_output"
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors to understand misclassifications.

        Args:
            X: Feature matrix
            y_true: True labels
            save_plots: Whether to save plots to disk
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing error analysis
        """
        log.info("Analyzing prediction errors...")

        # Get predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        # Identify errors
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]

        # Calculate confidence scores
        max_proba = np.max(y_pred_proba, axis=1)

        # Create analysis
        error_analysis = {
            'total_samples': len(X),
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(X),
            'error_indices': error_indices,
            'error_confidences': max_proba[error_indices],
            'correct_confidences': max_proba[~errors],
        }

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Error rate by class
        error_by_class = pd.crosstab(y_true, errors, normalize='index')
        error_by_class.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Error Rate by True Class')
        axes[0, 0].set_xlabel('True Class')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].legend(['Correct', 'Error'])

        # 2. Confidence distribution for errors vs correct
        axes[0, 1].hist(error_analysis['error_confidences'], bins=30, alpha=0.7, label='Errors', color='red')
        axes[0, 1].hist(error_analysis['correct_confidences'], bins=30, alpha=0.7, label='Correct', color='green')
        axes[0, 1].set_xlabel('Prediction Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Distribution: Errors vs Correct')
        axes[0, 1].legend()

        # 3. Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('True')

        # 4. Feature importance for errors vs correct
        if len(error_indices) > 0:
            # This would require more sophisticated analysis
            # For now, just show a placeholder
            axes[1, 1].text(
                0.5,
                0.5,
                'Feature importance\nanalysis for errors\nwould go here',
                ha='center',
                va='center',
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title('Error Analysis (Placeholder)')

        plt.tight_layout()

        # Save plots if requested
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'prediction_error_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"Prediction error analysis plot saved to: {plot_path}")

        log.info(f"Error analysis completed. Error rate: {error_analysis['error_rate']:.3f}")

        return error_analysis

    def comprehensive_analysis(
        self, X: pd.DataFrame, y_true: Optional[pd.Series] = None, output_dir: str = "analysis_output"
    ) -> Dict[str, Any]:
        """
        Run a comprehensive analysis using all available methods.

        Args:
            X: Feature matrix for analysis
            y_true: Optional true labels for error analysis
            output_dir: Directory to save all outputs

        Returns:
            Dictionary containing all analysis results
        """
        log.info("Starting comprehensive model interpretability analysis...")

        results = {}

        # 1. Feature Importance Analysis
        log.info("=" * 50)
        log.info("1. FEATURE IMPORTANCE ANALYSIS")
        log.info("=" * 50)
        results['feature_importance'] = self.analyze_feature_importance(top_n=20, save_plot=True, output_dir=output_dir)

        # 2. Tree Structure Analysis
        log.info("=" * 50)
        log.info("2. TREE STRUCTURE ANALYSIS")
        log.info("=" * 50)
        results['tree_structure'] = self.analyze_tree_structure(
            tree_index=0, max_depth=3, save_text=True, output_dir=output_dir
        )

        # 3. SHAP Analysis
        if SHAP_AVAILABLE:
            log.info("=" * 50)
            log.info("3. SHAP ANALYSIS")
            log.info("=" * 50)
            results['shap'] = self.analyze_shap_values(X=X, sample_size=100, save_plots=True, output_dir=output_dir)
        else:
            log.warning("Skipping SHAP analysis - not available")

        # 4. LIME Analysis
        if LIME_AVAILABLE:
            log.info("=" * 50)
            log.info("4. LIME ANALYSIS")
            log.info("=" * 50)
            results['lime'] = self.analyze_lime_explanations(
                X=X, sample_indices=list(range(min(5, len(X)))), save_plots=True, output_dir=output_dir
            )
        else:
            log.warning("Skipping LIME analysis - not available")

        # 5. Error Analysis (if true labels provided)
        if y_true is not None:
            log.info("=" * 50)
            log.info("5. PREDICTION ERROR ANALYSIS")
            log.info("=" * 50)
            results['error_analysis'] = self.analyze_prediction_errors(
                X=X, y_true=y_true, save_plots=True, output_dir=output_dir
            )
        else:
            log.warning("Skipping error analysis - no true labels provided")

        log.info("=" * 50)
        log.info("COMPREHENSIVE ANALYSIS COMPLETED")
        log.info("=" * 50)
        log.info(f"All outputs saved to: {output_dir}")

        return results


def load_model_and_analyze(
    model_path: str,
    feature_names: List[str],
    X: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    output_dir: str = "analysis_output",
) -> Dict[str, Any]:
    """
    Convenience function to load a model and run comprehensive analysis.

    Args:
        model_path: Path to the saved model file
        feature_names: List of feature names used in the model
        X: Feature matrix for analysis
        y_true: Optional true labels for error analysis
        output_dir: Directory to save all outputs

    Returns:
        Dictionary containing all analysis results
    """
    log.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    analyzer = ModelInterpretabilityAnalyzer(model, feature_names)
    results = analyzer.comprehensive_analysis(X, y_true, output_dir)

    return results


# Example usage and testing
if __name__ == "__main__":
    # This would be used with actual model and data
    print("Model Interpretability Analysis Tool")
    print("This module provides comprehensive analysis tools for understanding model predictions.")
    print("\nUsage:")
    print("1. Initialize analyzer: analyzer = ModelInterpretabilityAnalyzer(model, feature_names)")
    print("2. Run analysis: results = analyzer.comprehensive_analysis(X, y_true)")
    print("3. Or use convenience function: results = load_model_and_analyze(model_path, feature_names, X, y_true)")
