# ============================================================================
# NETWORK ROUTER FAILURE PREDICTION — END-TO-END PIPELINE
# EDA · Preprocessing · Model Training · Calibration · Threshold Selection
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress the seaborn deprecation warning to keep the console output clean
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load the dataset
df = pd.read_csv('datos_routers.csv')

# ============================================================================
# PROFESSIONAL STYLE CONFIGURATION
# ============================================================================
def set_professional_style():
    """Set a clean, portfolio-ready Matplotlib style for all charts."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

set_professional_style()

# Create the single output directory for all visualizations and artifacts
os.makedirs('outputs', exist_ok=True)

# Features used across the entire EDA and modeling pipeline
features = ['active_sessions', 'crc_errors_per_second', 'buffer_memory_utilization_percent',
            'unplanned_restarts_last_24h', 'dropped_packets_due_to_buffer_full_last_hour']


# ============================================================================
# 1. FEATURE DISTRIBUTIONS — HISTOGRAM / KDE / BAR CHARTS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Distribution Analysis of Network Metrics', fontsize=18, weight='bold', y=1.02)

# Distinct color per feature taken from the viridis colormap
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(features)))

def is_discrete_low_cardinality(series, max_unique=10):
    """Return True when the variable is discrete with only a few unique values."""
    return series.dtype in ['int64', 'int32', 'float64'] and series.nunique() <= max_unique

for idx, (ax, col, color) in enumerate(zip(axes.flatten(), features, colors)):
    data = df[col].dropna()
    unique_vals = data.nunique()
    is_low_cardinality = is_discrete_low_cardinality(data)

    if is_low_cardinality:
        # BAR CHART for discrete variables with few distinct values (e.g. 0-5)
        value_counts = data.value_counts().sort_index()
        bars = ax.bar(value_counts.index.astype(str), value_counts.values,
                      color=color, edgecolor='black', linewidth=1.2, alpha=0.8)

        # Annotate each bar with its absolute frequency
        for bar, count in zip(bars, value_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=9, weight='bold')

        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)

        # Summary statistics box (mean and median, no skewness for discrete vars)
        mean_val = data.mean()
        median_val = data.median()
        stats_text = f'μ={mean_val:.2f} | Median={median_val:.0f} '
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    else:
        # HISTOGRAM + KDE for continuous or high-cardinality variables
        n, bins, patches = ax.hist(data, bins=40, edgecolor='white',
                                   alpha=0.7, color=color, density=True, linewidth=0.5)

        # Overlay a kernel density estimate to visualize the smooth distribution
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'k-', linewidth=2)

        # Reference lines for the central tendency
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.2f}')

        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # Skewness annotation (only meaningful for continuous variables)
        skewness = data.skew()
        ax.text(0.98, 0.05, f'Skewness: {skewness:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Common styling applied to both chart types
    ax.set_title(f"{col.replace('_', ' ').title()}\n({unique_vals} unique values)",
                 fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Force clean integer ticks for low-cardinality variables
    if is_low_cardinality:
        ax.set_xticks(range(len(value_counts.index)))
        ax.set_xticklabels(value_counts.index.astype(int))

# Hide the unused sixth subplot (5 features, 6 axes)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/01_distributions_professional.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# 2. CORRELATION MATRIX — TRIANGULAR HEATMAP
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 9))

# Compute the pairwise correlation matrix and mask the upper triangle
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Diverging palette centered at zero for positive/negative correlations
cmap = sns.diverging_palette(230, 20, as_cmap=True, s=90, l=50)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            annot_kws={'size': 9, 'weight': 'bold'}, ax=ax)

ax.set_title('Feature Correlation Matrix\n(Upper Triangle)', fontsize=16, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Highlight strong correlations (|r| > 0.7) with a gold border
for i in range(len(corr_matrix.columns)):
    for j in range(i):  # Only inspect the lower triangle
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='gold', linewidth=2))

plt.tight_layout()
plt.savefig('outputs/02_correlation_matrix_professional.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# 3. TARGET CLASS IMBALANCE — ENHANCED BAR CHART
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Absolute counts and percentages for each target class
class_counts = df['target'].value_counts()
class_percent = df['target'].value_counts(normalize=True) * 100
labels = ['No Failure (0)', 'Failure (1)']
colors_bar = ['#2E86AB', '#A23B72']

# Bars representing each class with a subtle transparency
bars = ax.bar(labels, class_counts.values, color=colors_bar,
              edgecolor='black', linewidth=1.5, alpha=0.85, width=0.6)

# Add absolute count and percentage labels to each bar
for bar, count, pct in zip(bars, class_counts.values, class_percent.values):
    height = bar.get_height()
    # Absolute sample count (above the bar)
    ax.text(bar.get_x() + bar.get_width()/2., height + max(class_counts.values)*0.02,
            f'{count:,} samples', ha='center', va='bottom',
            fontsize=13, weight='bold', color='black')
    # Percentage of total (inside the bar)
    ax.text(bar.get_x() + bar.get_width()/2., height - max(class_counts.values)*0.08,
            f'({pct:.1f}%)', ha='center', va='top',
            fontsize=11, weight='normal', color='white')

# Imbalance ratio annotation (failure vs. no-failure)
imbalance_ratio = class_counts.values[1] / class_counts.values[0] if class_counts.values[0] > 0 else 0
ax.text(0.98, 0.95, f'Imbalance Ratio: {imbalance_ratio:.3f}\n(1:{1/imbalance_ratio:.1f})',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray'))

# Axis labels and title
ax.set_ylabel('Number of Samples', fontsize=12, weight='bold')
ax.set_xlabel('Target Class', fontsize=12, weight='bold')
ax.set_title('Target Class Distribution Analysis', fontsize=16, weight='bold', pad=20)

# Reference line showing what a perfectly balanced distribution would be
balanced_line = len(df) / 2
ax.axhline(y=balanced_line, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(len(labels)-0.5, balanced_line + max(class_counts.values)*0.02,
        f'Balanced (50%): {balanced_line:.0f} samples',
        ha='right', va='bottom', fontsize=9, color='gray', style='italic')

# Grid and spine styling
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)  # Keep the grid behind the bars
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add headroom above the tallest bar
ax.set_ylim(0, max(class_counts.values) * 1.2)

# Subtle background coloring for a polished look
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('outputs/03_class_imbalance_professional.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# 4. FEATURE–TARGET RELATIONSHIPS — VIOLIN / BOX / GROUPED BARS
# ============================================================================
# Consistent color scheme shared across all subplots
colors_box = {'No Failure': '#2E86AB', 'Failure': '#A23B72'}

# Figure with 2 rows and 3 columns (6 subplots for 5 features)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Feature Distribution by Target Class', fontsize=18, weight='bold', y=1.02)

# Flatten axes for straightforward iteration
axes_flat = axes.flatten()

for idx, (ax, col) in enumerate(zip(axes_flat, features)):

    # ----- Special case: discrete variable with values 0-5 -----
    if col == 'unplanned_restarts_last_24h':
        # Build a contingency table (values 0..5 vs. target class)
        # Ensure every value 0..5 is present, even with zero frequency
        all_vals = np.arange(0, 6)
        counts = []
        for target_val in [0, 1]:
            # Frequency of each value 0-5 within this class
            freq = df[df['target'] == target_val][col].value_counts().reindex(all_vals, fill_value=0)
            counts.append(freq.values)

        # Grouped bar configuration
        x = np.arange(len(all_vals))  # x positions for values 0..5
        width = 0.35  # width of each individual bar

        # Bars for the No-Failure class (target = 0)
        bars0 = ax.bar(x - width/2, counts[0], width, label='No Failure (0)',
                       color=colors_box['No Failure'], edgecolor='black', linewidth=1)
        # Bars for the Failure class (target = 1)
        bars1 = ax.bar(x + width/2, counts[1], width, label='Failure (1)',
                       color=colors_box['Failure'], edgecolor='black', linewidth=1)

        # Annotate each non-zero bar with its count
        for bars in [bars0, bars1]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts[0].max(), counts[1].max())*0.02,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)

        # Labels and formatting
        ax.set_xticks(x)
        ax.set_xticklabels(all_vals)
        ax.set_xlabel('Number of unplanned restarts', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Unplanned Restarts Last 24H', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        # ax.legend(loc='upper right', fontsize=8)  # optional, a global legend is used instead

        # Class totals (kept for potential statistical annotation)
        total0 = counts[0].sum()
        total1 = counts[1].sum()

    # ----- Standard case: continuous variables -----
    else:
        data_to_plot = [df[df['target'] == 0][col].dropna(),
                        df[df['target'] == 1][col].dropna()]

        # Violin plot to show the full distribution shape
        parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=False, showmedians=True)
        for pc, color in zip(parts['bodies'], colors_box.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')

        # Boxplot overlay to expose quartiles and outliers
        bp = ax.boxplot(data_to_plot, positions=[1, 2], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))

        for box, color in zip(bp['boxes'], colors_box.values()):
            box.set_facecolor(color)
            box.set_alpha(0.3)

        # Per-class statistical annotations (n, mean, std)
        for i, target_val in enumerate([0, 1]):
            data = df[df['target'] == target_val][col]
            stats_text = f'n={len(data)}\nμ={data.mean():.2f}\nσ={data.std():.2f}'
            ax.text(i+1, ax.get_ylim()[1] * 0.95, stats_text, ha='center', va='top',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_title(col.replace('_', ' ').title(), fontsize=12, weight='bold')
        ax.set_xlabel('Target Class', fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Failure (0)', 'Failure (1)'], rotation=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Hide the unused sixth subplot (5 features, 6 axes)
axes_flat[-1].axis('off')

# Shared legend for every subplot
legend_elements = [Patch(facecolor=colors_box['No Failure'], alpha=0.9, label='No Failure'),
                   Patch(facecolor=colors_box['Failure'], alpha=0.9, label='Failure')]
fig.legend(handles=legend_elements, loc='lower right', ncol=2, fontsize=11,
           frameon=True, shadow=True, title='Target Class')

plt.tight_layout()
plt.savefig('outputs/04_feature_target_relationships_professional.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# EDA SUMMARY REPORT
# ============================================================================
print("="*60)
print("PROFESSIONAL EDA COMPLETED SUCCESSFULLY")
print("="*60)
print("\n📊 Generated Visualizations:")
print("  1. Distributions with KDE and summary statistics")
print("  2. Triangular correlation matrix with highlights")
print("  3. Target class imbalance bar chart")
print("  4. Violin + boxplot for feature-target analysis")
print(f"\n📁 All figures saved to: {os.path.abspath('outputs/')}")
print("\n💡 Pro tip: Embed these images in your Notion portfolio with markdown:")
print("  ```markdown")
print("  ![Distribution Analysis](./outputs/01_distributions_professional.png)")
print("  ```")
print("="*60)


# ============================================================================
# PREPROCESSING — TRAIN/TEST SPLIT AND FEATURE SCALING
# (preprocessing.py — executed directly in the VSCode terminal)
# ============================================================================

# Create directories for processed data and serialized models
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Dataset loaded: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Target distribution — Full dataset:")
print(f"  No Failure (0): {len(y[y==0]):,} ({len(y[y==0])/len(y)*100:.2f}%)")
print(f"  Failure (1):    {len(y[y==1]):,} ({len(y[y==1])/len(y)*100:.2f}%)")

# Stratified train/test split (80/20, preserving the class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nStratified Split Results:")
print(f"  Training set: {X_train.shape[0]:,} rows ({X_train.shape[0]/X.shape[0]*100:.0f}%)")
print(f"  Test set:     {X_test.shape[0]:,} rows ({X_test.shape[0]/X.shape[0]*100:.0f}%)")
print(f"\nTraining target distribution:")
print(f"  No Failure (0): {len(y_train[y_train==0]):,} ({len(y_train[y_train==0])/len(y_train)*100:.2f}%)")
print(f"  Failure (1):    {len(y_train[y_train==1]):,} ({len(y_train[y_train==1])/len(y_train)*100:.2f}%)")
print(f"\nTest target distribution:")
print(f"  No Failure (0): {len(y_test[y_test==0]):,} ({len(y_test[y_test==0])/len(y_test)*100:.2f}%)")
print(f"  Failure (1):    {len(y_test[y_test==1]):,} ({len(y_test[y_test==1])/len(y_test)*100:.2f}%)")

# Feature scaling (fit only on the training data to avoid information leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify that scaled training features have ~0 mean and ~1 std
print(f"\nFeature Scaling Verification (first 3 features):")
feature_names = X.columns.tolist()
for i in range(min(3, len(feature_names))):
    print(f"  {feature_names[i]:<45} — Train mean: {X_train_scaled[:, i].mean():.4f}, "
          f"Train std: {X_train_scaled[:, i].std():.4f}")

# Persist the processed arrays for downstream steps
np.save('data/X_train_scaled.npy', X_train_scaled)
np.save('data/X_test_scaled.npy', X_test_scaled)
np.save('data/y_train.npy', y_train.values)
np.save('data/y_test.npy', y_test.values)

# Persist the fitted scaler for future inference
joblib.dump(scaler, 'models/scaler.pkl')

# Persist feature names for interpretability later in the pipeline
pd.Series(X.columns).to_csv('data/feature_names.csv', index=False)

print(f"\n✓ Preprocessing complete. Files saved:")
print(f"  → data/X_train_scaled.npy")
print(f"  → data/X_test_scaled.npy")
print(f"  → data/y_train.npy")
print(f"  → data/y_test.npy")
print(f"  → models/scaler.pkl")
print(f"  → data/feature_names.csv")


# ============================================================================
# MODEL TRAINING — LOGISTIC REGRESSION · RANDOM FOREST · XGBOOST
# (model_training.py — executed directly in the VSCode terminal)
# ============================================================================
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, brier_score_loss

# Load the preprocessed data produced in the previous step
print("\n" + "█" * 70)
print("LOADING PREPROCESSED DATA")
print("█" * 70)
X_train = np.load('data/X_train_scaled.npy')
X_test = np.load('data/X_test_scaled.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
feature_names = pd.read_csv('data/feature_names.csv').iloc[:, 0].tolist()

print(f"  Training set: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"  Test set:     {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
print(f"  Positive class (train): {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Positive class (test):  {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# 5-fold stratified cross-validation, shared across all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom scorer: GridSearchCV maximizes, so we return the negative Brier score
def brier_score_scorer(estimator, X, y):
    """Return the negative Brier score so GridSearchCV minimizes calibration error."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return -brier_score_loss(y, y_prob)

# ████████████████████████████████████████████████████████████████████
# MODEL 1: LOGISTIC REGRESSION
# ████████████████████████████████████████████████████████████████████
print("\n" + "=" * 70)
print("  MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'),
    param_grid=lr_param_grid,
    cv=cv,
    scoring=brier_score_scorer,
    n_jobs=-1,
    verbose=0
)
lr_grid.fit(X_train, y_train)

# Cross-validation results for every regularization strength
print("\n  ── Cross-Validation Results ──")
cv_results_lr = pd.DataFrame(lr_grid.cv_results_)
for _, row in cv_results_lr.iterrows():
    print(f"  C={row['param_C']:<8}  Mean Brier: {-row['mean_test_score']:.4f}  (±{row['std_test_score']:.4f})")
print(f"\n  ► Best: C={lr_grid.best_params_['C']}  |  Best CV Brier: {-lr_grid.best_score_:.4f}")

# ████████████████████████████████████████████████████████████████████
# MODEL 2: RANDOM FOREST
# ████████████████████████████████████████████████████████████████████
print("\n" + "=" * 70)
print("  MODEL 2: RANDOM FOREST")
print("=" * 70)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [10, 20, 50]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_grid=rf_param_grid,
    cv=cv,
    scoring=brier_score_scorer,
    n_jobs=1,  # Set to 1 to avoid nested parallelization conflicts with the RF n_jobs
    verbose=0
)
rf_grid.fit(X_train, y_train)

# Show the top 5 hyperparameter combinations by Brier score
print("\n  ── Top 5 Cross-Validation Results ──")
cv_results_rf = pd.DataFrame(rf_grid.cv_results_)
cv_results_rf['mean_brier'] = -cv_results_rf['mean_test_score']
top5_rf = cv_results_rf.nsmallest(5, 'mean_brier')
for i, (_, row) in enumerate(top5_rf.iterrows(), 1):
    print(f"  #{i}  n_est={int(row['param_n_estimators']):<5}  depth={str(row['param_max_depth']):<6}  "
          f"min_leaf={int(row['param_min_samples_leaf']):<4}  Brier: {row['mean_brier']:.4f}  (±{row['std_test_score']:.4f})")
print(f"\n  ► Best: n_estimators={rf_grid.best_params_['n_estimators']}, "
      f"max_depth={rf_grid.best_params_['max_depth']}, "
      f"min_samples_leaf={rf_grid.best_params_['min_samples_leaf']}")
print(f"  ► Best CV Brier: {-rf_grid.best_score_:.4f}")

# ████████████████████████████████████████████████████████████████████
# MODEL 3: XGBOOST
# ████████████████████████████████████████████████████████████████████
print("\n" + "=" * 70)
print("  MODEL 3: XGBOOST")
print("=" * 70)

# Weight positive class inversely to its frequency to handle the imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    ),
    param_grid=xgb_param_grid,
    cv=cv,
    scoring=brier_score_scorer,
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_train, y_train)

# Show the top 5 hyperparameter combinations by Brier score
print("\n  ── Top 5 Cross-Validation Results ──")
cv_results_xgb = pd.DataFrame(xgb_grid.cv_results_)
cv_results_xgb['mean_brier'] = -cv_results_xgb['mean_test_score']
top5_xgb = cv_results_xgb.nsmallest(5, 'mean_brier')
for i, (_, row) in enumerate(top5_xgb.iterrows(), 1):
    print(f"  #{i}  n_est={int(row['param_n_estimators']):<5}  depth={int(row['param_max_depth']):<4}  "
          f"lr={row['param_learning_rate']:<6}  Brier: {row['mean_brier']:.4f}  (±{row['std_test_score']:.4f})")
print(f"\n  ► Best: n_estimators={xgb_grid.best_params_['n_estimators']}, "
      f"max_depth={xgb_grid.best_params_['max_depth']}, "
      f"learning_rate={xgb_grid.best_params_['learning_rate']}")
print(f"  ► Best CV Brier: {-xgb_grid.best_score_:.4f}")

# ████████████████████████████████████████████████████████████████████
# FINAL MODEL COMPARISON ON THE HELD-OUT TEST SET
# ████████████████████████████████████████████████████████████████████
print("\n" + "█" * 70)
print("  FINAL MODEL COMPARISON ON TEST SET")
print("█" * 70)

models = {
    'Logistic Regression': lr_grid.best_estimator_,
    'Random Forest':       rf_grid.best_estimator_,
    'XGBoost':             xgb_grid.best_estimator_
}

# Compute a full battery of metrics for every tuned model
results = []
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    ll = log_loss(y_test, y_prob)
    bs = brier_score_loss(y_test, y_prob)

    # Classification metrics computed at the default 0.5 threshold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Log-Loss': ll,
        'Brier Score': bs,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC-ROC': auc
    })

# Render the comparison as a formatted ASCII table
print("\n  ┌─────────────────────────────┬──────────┬─────────────┬──────────┬───────────┬────────┬──────────┬─────────┐")
print("  │ Model                       │ Log-Loss │ Brier Score │ Accuracy │ Precision │ Recall │ F1 Score │ AUC-ROC │")
print("  ├─────────────────────────────┼──────────┼─────────────┼──────────┼───────────┼────────┼──────────┼─────────┤")
for r in results:
    print(f"  │ {r['Model']:<27} │ {r['Log-Loss']:.4f}   │ {r['Brier Score']:.4f}     │ "
          f"{r['Accuracy']:.4f}  │ {r['Precision']:.4f}   │ {r['Recall']:.4f} │ {r['F1 Score']:.4f}  │ {r['AUC-ROC']:.4f} │")
print("  └─────────────────────────────┴──────────┴─────────────┴──────────┴───────────┴────────┴──────────┴─────────┘")

# Identify the best model by lowest Brier score (best calibration)
best_idx = np.argmin([r['Brier Score'] for r in results])
print(f"\n  ► BEST MODEL: {results[best_idx]['Model']} (lowest Brier Score: {results[best_idx]['Brier Score']:.4f})")

# Feature importance ranking for the Random Forest model
print("\n" + "─" * 70)
print("  FEATURE IMPORTANCE (Best Tree-Based Model)")
print("─" * 70)
best_tree_model = models[results[1]['Model']]  # Random Forest
if hasattr(best_tree_model, 'feature_importances_'):
    importances = best_tree_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices):
        bar = "█" * int(importances[idx] * 50)
        print(f"  {i+1}. {feature_names[idx]:<45} {importances[idx]:.4f}  {bar}")

# Persist every tuned best estimator
joblib.dump(lr_grid.best_estimator_, 'models/logistic_regression_best.pkl')
joblib.dump(rf_grid.best_estimator_, 'models/random_forest_best.pkl')
joblib.dump(xgb_grid.best_estimator_, 'models/xgboost_best.pkl')

# Persist the comparison table to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/model_comparison.csv', index=False)

print("\n" + "█" * 70)
print("  TRAINING COMPLETE")
print("█" * 70)
print("  Best models saved to /models folder")
print("  Comparison results saved to outputs/model_comparison.csv")
print("█" * 70 + "\n")


# ============================================================================
# PROBABILITY CALIBRATION — RANDOM FOREST & XGBOOST
# (calibration.py — executed directly in the VSCode terminal)
# ============================================================================
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("\n" + "█" * 70)
print("  PROBABILITY CALIBRATION (Random Forest & XGBoost)")
print("█" * 70)

# 1. Load the preprocessed data
X_train = np.load('data/X_train_scaled.npy')
X_test = np.load('data/X_test_scaled.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
print(f"\n  Training set: {X_train.shape[0]:,} rows")
print(f"  Test set:     {X_test.shape[0]:,} rows")
print(f"  Positive class (train): {y_train.mean()*100:.1f}%")
print(f"  Positive class (test):  {y_test.mean()*100:.1f}%")

# 2. Split the training set into a fit set and a calibration validation set
X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
print(f"\n  Calibration split:")
print(f"    Cal training:  {X_cal_train.shape[0]:,} rows (positive: {y_cal_train.mean()*100:.1f}%)")
print(f"    Cal validation: {X_cal_val.shape[0]:,} rows (positive: {y_cal_val.mean()*100:.1f}%)")

# 3. Load the previously trained base models
print("\n" + "─" * 70)
print("  LOADING BASE MODELS")
print("─" * 70)

rf_base = joblib.load('models/random_forest_best.pkl')
xgb_base = joblib.load('models/xgboost_best.pkl')
print("  ✓ Random Forest loaded")
print("  ✓ XGBoost loaded")

models_to_calibrate = {
    "Random Forest": rf_base,
    "XGBoost": xgb_base
}

# Calibration methods to evaluate (None = baseline / uncalibrated)
methods = {
    'Uncalibrated': None,
    'Platt Scaling': 'sigmoid',
    'Isotonic Regression': 'isotonic'
}

# Store all per-model results in a dictionary
all_results = {}

for model_name, base_model in models_to_calibrate.items():
    print("\n" + "=" * 70)
    print(f"  CALIBRATING {model_name.upper()}")
    print("=" * 70)

    # Refit the base model on the calibration training subset
    base_model.fit(X_cal_train, y_cal_train)

    prob_vectors = {}
    # Baseline uncalibrated probabilities
    prob_vectors['Uncalibrated'] = base_model.predict_proba(X_test)[:, 1]

    # Apply each calibration method using the held-out calibration validation set
    for method_name, method_type in list(methods.items())[1:]:
        print(f"  → Applying {method_name}...")
        calibrated_clf = CalibratedClassifierCV(base_model, method=method_type, cv='prefit')
        calibrated_clf.fit(X_cal_val, y_cal_val)
        prob_vectors[method_name] = calibrated_clf.predict_proba(X_test)[:, 1]
        # Persist the calibrated model
        joblib.dump(calibrated_clf, f'models/{model_name.lower().replace(" ", "_")}_{method_type}.pkl')

    # Compute calibration metrics for each probability vector
    results = {'method': [], 'brier': [], 'ece': [], 'log_loss': []}
    for method_name, probs in prob_vectors.items():
        bs = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        # Expected Calibration Error (ECE) over 10 equal-width bins
        bin_edges = np.linspace(0, 1, 11)
        bin_indices = np.digitize(probs, bin_edges) - 1
        ece = 0.0
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                ece += (mask.sum() / len(probs)) * abs(probs[mask].mean() - y_test[mask].mean())
        results['method'].append(method_name)
        results['brier'].append(bs)
        results['ece'].append(ece)
        results['log_loss'].append(ll)

    # Print a per-model metrics table
    print("\n  ┌──────────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ Method                   │ Brier    │ ECE      │ Log-Loss │")
    print("  ├──────────────────────────┼──────────┼──────────┼──────────┤")
    for i in range(len(results['method'])):
        print(f"  │ {results['method'][i]:<24} │ {results['brier'][i]:.4f}   │ {results['ece'][i]:.4f}   │ {results['log_loss'][i]:.4f}   │")
    print("  └──────────────────────────┴──────────┴──────────┴──────────┘")

    # Identify the best calibration method by Brier score
    best_idx = np.argmin(results['brier'])
    best_method = results['method'][best_idx]
    best_brier = results['brier'][best_idx]
    improvement = results['brier'][0] - best_brier
    print(f"\n  ► Best: {best_method} (Brier = {best_brier:.4f})")
    print(f"  ► Improvement over uncalibrated: {improvement:.4f} ({improvement/results['brier'][0]*100:.1f}%)")

    # Store every artifact for this model in a structured dictionary
    all_results[model_name] = {
        'results': results,
        'prob_vectors': prob_vectors,
        'best_method': best_method,
        'best_probs': prob_vectors[best_method]
    }

# ----------------------------------------------------------------------------
# RELIABILITY DIAGRAMS (one figure per model, three calibration methods each)
# ----------------------------------------------------------------------------
print("\n" + "─" * 70)
print("  GENERATING RELIABILITY DIAGRAMS")
print("─" * 70)

for model_name, data in all_results.items():
    prob_vectors = data['prob_vectors']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ordered_methods = ['Uncalibrated', 'Platt Scaling', 'Isotonic Regression']
    for ax, method in zip(axes, ordered_methods):
        if method not in prob_vectors:
            continue
        y_prob = prob_vectors[method]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
        bs = brier_score_loss(y_test, y_prob)
        ll = log_loss(y_test, y_prob)
        # Expected Calibration Error for the annotation
        bin_edges = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        ece = 0.0
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_test[mask].mean())
        # Diagonal reference, model curve, and per-bin error segments
        ax.plot([0,1], [0,1], 'k--', label='Perfect', alpha=0.5)
        ax.plot(prob_pred, prob_true, 's-', color='#3498db', linewidth=2.5, markersize=8, label='Model')
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color='#3498db')
        for px, py in zip(prob_pred, prob_true):
            ax.plot([px, px], [px, py], color='#e74c3c', linewidth=0.8, alpha=0.4)
        ax.set_title(f'{method}\nBrier: {bs:.4f}  |  ECE: {ece:.4f}  |  Log-Loss: {ll:.4f}', fontsize=11)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
    plt.suptitle(f'Reliability Diagram: {model_name}', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/calibration_{model_name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  → Saved: outputs/calibration_{model_name.lower().replace(' ', '_')}.png")

# ----------------------------------------------------------------------------
# PROBABILITY DISTRIBUTION COMPARISON (uncalibrated vs. best calibrated)
# ----------------------------------------------------------------------------
print("\n" + "─" * 70)
print("  PROBABILITY DISTRIBUTION COMPARISON")
print("─" * 70)

for model_name, data in all_results.items():
    prob_vectors = data['prob_vectors']
    best_method = data['best_method']
    uncal_probs = prob_vectors['Uncalibrated']
    cal_probs = prob_vectors[best_method]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, probs, title in zip(axes, [uncal_probs, cal_probs], ['Uncalibrated', f'Calibrated ({best_method})']):
        ax.hist(probs[y_test==0], bins=30, alpha=0.6, label='No Failure (0)', color='#2ecc71', edgecolor='black')
        ax.hist(probs[y_test==1], bins=30, alpha=0.6, label='Failure (1)', color='#e74c3c', edgecolor='black')
        ax.set_title(f'{model_name} — {title}', fontsize=11, weight='bold')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/prob_dist_{model_name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  → Saved: outputs/prob_dist_{model_name.lower().replace(' ', '_')}.png")


# ============================================================================
# THRESHOLD SELECTION FOR MAINTENANCE ALERTS
# (threshold_selection.py — executed directly in the VSCode terminal)
# ============================================================================
from sklearn.metrics import brier_score_loss, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve

print("\n" + "█" * 70)
print("  THRESHOLD SELECTION FOR MAINTENANCE ALERTS")
print("█" * 70)

# -------------------------------------------------------------------
# 1. Load the test data and the best calibrated model
# -------------------------------------------------------------------
X_test = np.load('data/X_test_scaled.npy')
y_test = np.load('data/y_test.npy')
best_model = joblib.load('models/random_forest_isotonic.pkl')
print(f"Test set: {len(y_test):,} rows (positive: {y_test.mean()*100:.1f}%)")

# -------------------------------------------------------------------
# 2. Predict failure probabilities on the test set
# -------------------------------------------------------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, y_prob)
logloss = log_loss(y_test, y_prob)

# -------------------------------------------------------------------
# 3. Calibration curve & Expected Calibration Error (ECE)
# -------------------------------------------------------------------
n_bins = 10
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy='uniform')
bin_edges = np.linspace(0, 1, n_bins+1)
bin_indices = np.digitize(y_prob, bin_edges) - 1
ece = 0.0
for i in range(n_bins):
    mask = bin_indices == i
    if mask.sum() > 0:
        ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_test[mask].mean())

# Plot the calibration (reliability) curve
plt.figure(figsize=(7,5))
plt.plot([0,1], [0,1], 'k--', label='Perfect')
plt.plot(prob_pred, prob_true, 's-', color='#2E86AB', lw=2.5, label='Calibrated RF (Isotonic)')
plt.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color='#2E86AB')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title(f'Calibration Curve (ECE = {ece:.4f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/calibration_curve_threshold.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 4. Gain & Lift charts (business value of risk-based targeting)
# -------------------------------------------------------------------
df = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
df['cum_pop_pct'] = (np.arange(1, len(df)+1) / len(df)) * 100
df['cum_pos'] = df['y_true'].cumsum()
total_pos = df['y_true'].sum()
df['gain'] = (df['cum_pos'] / total_pos) * 100
df['lift'] = df['gain'] / df['cum_pop_pct']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.plot(df['cum_pop_pct'], df['gain'], lw=2.5, color='#2E86AB', label='Model')
ax1.plot([0,100], [0,100], 'k--', alpha=0.5, label='Random')
ax1.set_xlabel('Population (%) sorted by risk')
ax1.set_ylabel('Failures captured (%)')
ax1.set_title('Cumulative Gain Chart')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(df['cum_pop_pct'], df['lift'], lw=2.5, color='#A23B72', label='Lift')
ax2.axhline(1, color='k', ls='--', alpha=0.5)
ax2.set_xlabel('Population (%) sorted by risk')
ax2.set_ylabel('Lift')
ax2.set_title('Lift Chart')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/gain_lift_charts.png', dpi=150)
plt.show()

# Business interpretation: failures captured by acting on the top risk deciles
gain_at_20 = df[df['cum_pop_pct'] <= 20]['gain'].max()
gain_at_30 = df[df['cum_pop_pct'] <= 30]['gain'].max()
print(f"\nBusiness gain: acting on top 20% risk → capture {gain_at_20:.1f}% of failures")
print(f"               acting on top 30% risk → capture {gain_at_30:.1f}% of failures")

# -------------------------------------------------------------------
# 5. Cost-based threshold optimisation
# -------------------------------------------------------------------
cost_fp = 1      # cost of a false alarm (unnecessary redundancy)
cost_fn = 10     # cost of a missed failure (much higher than a false alarm)

thresholds = np.linspace(0.01, 0.99, 100)
total_costs = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    cost = fp * cost_fp + fn * cost_fn
    total_costs.append(cost)

# Threshold that minimizes the total expected cost
opt_idx = np.argmin(total_costs)
opt_thresh = thresholds[opt_idx]
min_cost = total_costs[opt_idx]

plt.figure(figsize=(8,5))
plt.plot(thresholds, total_costs, lw=2, color='#e74c3c')
plt.axvline(opt_thresh, color='k', ls='--', label=f'Optimal threshold = {opt_thresh:.3f}')
plt.xlabel('Decision threshold')
plt.ylabel('Total cost (test set)')
plt.title('Cost-based threshold optimisation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/cost_vs_threshold.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 6. Performance at the optimal threshold
# -------------------------------------------------------------------
y_pred_opt = (y_prob >= opt_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt, labels=[0,1]).ravel()
precision = tp / (tp+fp) if (tp+fp)>0 else 0
recall = tp / (tp+fn) if (tp+fn)>0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision+recall)>0 else 0

print(f"\nOptimal threshold = {opt_thresh:.3f}")
print(f"Precision = {precision:.4f} | Recall = {recall:.4f} | F1 = {f1:.4f}")
print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# -------------------------------------------------------------------
# 7. Save the summary results
# -------------------------------------------------------------------
summary = pd.DataFrame({
    'Metric': ['Brier', 'Log-Loss', 'ECE', 'Optimal threshold',
               'Precision (optimal)', 'Recall (optimal)', 'F1 (optimal)',
               'Gain @20%', 'Gain @30%'],
    'Value': [f'{brier:.4f}', f'{logloss:.4f}', f'{ece:.4f}', f'{opt_thresh:.3f}',
              f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}',
              f'{gain_at_20:.1f}%', f'{gain_at_30:.1f}%']
})
summary.to_csv('outputs/threshold_analysis_summary.csv', index=False)
print("\n✓ Summary saved to outputs/threshold_analysis_summary.csv")
print("\n" + "█" * 70 + "\n")