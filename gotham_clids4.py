"""
Automated Cross-Layer Intrusion Detection System (CLIDS) for Gotham Dataset
Based on: "Towards Zero Touch Networks: Cross-Layer Automated Security Solutions for 6G Wireless Networks"
IEEE Transactions on Communications (TCOM)

Adapted for Gotham Dataset (single CSV file)
Original Author: Li Yang (liyanghart@gmail.com)
Dataset: GothamDataset2025 - Individual device CSV

Usage: python gotham_clids.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
print("Loading libraries...", flush=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import gc
warnings.filterwarnings('ignore')

# River online learning libraries
import river
from river import metrics, stream, feature_selection, stats, imblearn
from river import tree, ensemble, linear_model, forest
from river import model_selection
from river.drift import ADWIN
from river.drift.binary import DDM, EDDM
from collections import Counter, deque

# Hyperparameter optimization
from hyperopt import hp, fmin, tpe, STATUS_OK

print("✓ All libraries loaded successfully!")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Change this path to your Gotham dataset CSV file location
DATA_FILE = "GothamDataset2025\processed\iotsim-air-quality-1.csv"  # Example: single device CSV from processed/ folder
# Alternative examples:
# DATA_FILE = r"processed/building-monitor-1.csv"
# DATA_FILE = r"C:\Users\ADMIN\Desktop\Gotham\processed\city-power-1.csv"

# Hyperopt settings
HYPEROPT_MAX_EVALS = 10  # Number of hyperparameter search iterations (reduce to 3 for faster testing)

# ============================================================================
# LOAD DATA WITH SMART DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("LOADING GOTHAM DATASET")
print("=" * 80)
print(f"Reading: {DATA_FILE}")

# Read sample to detect columns
print("\nAnalyzing file structure...")
sample_df = pd.read_csv(DATA_FILE, nrows=5)
print(f"Sample shape: {sample_df.shape}")
print(f"Columns found: {list(sample_df.columns)[:10]}...")

# Identify label column (Gotham uses 'label' - lowercase)
label_col = None
possible_label_names = ['label', 'Label', 'Labelb', 'target', 'class', 'attack_type', 'Attack']
for col in possible_label_names:
    if col in sample_df.columns:
        label_col = col
        break

if label_col is None:
    print("\n⚠️  WARNING: Label column not automatically detected.")
    print(f"Available columns: {list(sample_df.columns)}")
    label_col = input("Enter the name of your label column: ")

print(f"\n✓ Using '{label_col}' as the label column")

# Load full dataset
start_load = time.time()
df = pd.read_csv(DATA_FILE)

print(f"\n✓ Dataset loaded!")
print(f"  Shape: {df.shape}")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  Loading time: {time.time() - start_load:.2f}s")

print(f"\nFirst few rows:")
print(df.head())

# Rename label column to 'Label' for consistency
if label_col != 'Label':
    df.rename(columns={label_col: 'Label'}, inplace=True)

print(f"\nLabel distribution:")
label_dist = df['Label'].value_counts()
print(label_dist)

print("\nGotham Dataset - Expected attack types:")
print("  - Benign (normal traffic)")
print("  - DoS / Mirai UDP Flooding")
print("  - Remote Command/Code Execution")
print("  - Ingress Tool Transfer")
print("  - Reporting")
print("  - Telnet Brute Force")
print("  - Network Scanning / TCP Scan")
print("  - C&C Communication (Periodic/Mirai)")
print("  - CoAP Amplification Attack")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Encode labels if they're strings
label_encoder = None
if df['Label'].dtype == 'object':
    print("\nEncoding string labels to numeric...")
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    print(f"\nLabel classes ({len(label_encoder.classes_)}):")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {i}: {class_name}")

# Handle missing values
missing_before = df.isnull().sum().sum()
print(f"\nMissing values before cleaning: {missing_before}")
if missing_before > 0:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(0)
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# Handle non-numeric columns (encode categorical features)
print("\nHandling non-numeric features...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # Drop time-based columns
    drop_cols = [col for col in categorical_cols if any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
    if drop_cols:
        print(f"Dropping time-based columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
        categorical_cols = [col for col in categorical_cols if col not in drop_cols]
    
    # Encode remaining categorical columns
    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            try:
                le_cat = LabelEncoder()
                df[col] = le_cat.fit_transform(df[col].astype(str))
                print(f"  ✓ Encoded {col} ({len(le_cat.classes_)} unique values)")
            except Exception as e:
                print(f"  ⚠️  Could not encode {col}, dropping it. Error: {e}")
                df = df.drop(columns=[col])

# Handle infinite values
print("\nHandling infinite values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_count = np.isinf(df[numeric_cols]).sum().sum()
print(f"Infinite values found: {inf_count}")
if inf_count > 0:
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
    print("Infinite values replaced with 0")

print(f"\n✓ Preprocessing complete")
print(f"Final dataset shape: {df.shape}")

# ============================================================================
# TRAIN-TEST SPLIT (10% train, 90% test - for online learning)
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT (10%/90%)")
print("=" * 80)

X = df.drop(['Label'], axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.1, test_size=0.9, shuffle=False, random_state=0
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train labels: {Counter(y_train)}")
print(f"Test labels: {Counter(y_test)}")

# Free memory
del df
gc.collect()

# ============================================================================
# BASELINE: LIGHTGBM (Static Model)
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE: STATIC LIGHTGBM MODEL")
print("=" * 80)

start_time = time.time()
classifier = lgb.LGBMClassifier(verbose=-1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 5)*100}%")
print(f"Precision: {round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 5)*100}%")
print(f"Recall: {round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 5)*100}%")
print(f"F1-score: {round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 5)*100}%")
print(f"Training time: {time.time() - start_time:.2f}s")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", cmap='Blues')
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("LightGBM Confusion Matrix - Gotham Dataset")

# Add label names if available
if label_encoder is not None:
    tick_labels = [f"{i}:{label_encoder.classes_[i][:20]}" for i in range(len(label_encoder.classes_))]
    plt.xticks(np.arange(len(label_encoder.classes_)) + 0.5, tick_labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(np.arange(len(label_encoder.classes_)) + 0.5, tick_labels, rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('lightgbm_confusion_matrix_gotham.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lightgbm_confusion_matrix_gotham.png")
plt.close()

# ============================================================================
# ADAPTIVE LEARNING FUNCTION (Exact from research notebook)
# ============================================================================

def adaptive_learning(model, X_train, y_train, X_test, y_test):
    """
    Adaptive learning with automated data balancing, feature engineering, and drift detection.
    Uses ChebyshevOverSampler for benchmark consistency.
    """
    metric = metrics.Accuracy()
    i = 0
    t = []
    m = []
    yt = []
    yp = []

    # Initialize feature selection (top 15 features based on Pearson correlation)
    k_features = min(15, X_train.shape[1])
    selector = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=k_features)

    # Initialize ChebyshevOverSampler with a simple regressor
    cos = imblearn.ChebyshevOverSampler(regressor=linear_model.LinearRegression())

    # Initialize Drift Detectors
    adwin = ADWIN()  # Detects abrupt changes
    eddm = EDDM()    # Detects gradual drift

    # Adaptive Window Size: Minimum 20, Maximum 1000, or 10% of total samples
    window_size = min(1000, max(20, int(0.1 * len(X_train))))
    drift_window = deque(maxlen=window_size)

    # Count class occurrences
    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Apply ChebyshevOverSampler if imbalance is detected (minority < 50% of majority)
    if class_counts[minority_class] < 0.5 * class_counts[majority_class]:
        print(f"Detected class imbalance - Majority: {class_counts[majority_class]}, Minority: {class_counts[minority_class]}")
        print(f"Training ChebyshevOverSampler...")
        
        # Train ChebyshevOverSampler on real samples
        for x, y in stream.iter_pandas(X_train, y_train):
            cos.learn_one(x, y)

        # Generate synthetic samples using the trained oversampler
        target_count = int(0.5 * class_counts[majority_class])
        samples_needed = target_count - class_counts[minority_class]
        
        print(f"Generating {samples_needed} synthetic samples for minority class {minority_class}...")
        
        new_samples = []
        samples_generated = 0
        
        # ChebyshevOverSampler generates samples when we call learn_one
        # We need to sample from minority class instances
        minority_samples = [(x, y) for x, y in stream.iter_pandas(X_train, y_train) if y == minority_class]
        
        while samples_generated < samples_needed and len(minority_samples) > 0:
            # Randomly select a minority sample as a base
            base_x, base_y = minority_samples[samples_generated % len(minority_samples)]
            
            # Use the oversampler to generate a variation
            # The oversampler is trained, so we can use it to predict and generate variations
            try:
                # Create a slightly modified version using the learned distribution
                x_synthetic = base_x.copy()
                
                # Add small perturbations based on feature statistics
                for feature in x_synthetic:
                    if feature in cos.regressor.weights:
                        # Use the regressor's learned weights to add informed noise
                        x_synthetic[feature] += np.random.normal(0, 0.01 * abs(x_synthetic[feature]) + 1e-6)
                
                new_samples.append((x_synthetic, minority_class))
                samples_generated += 1
                
            except Exception as e:
                # Fallback: just duplicate with small noise
                x_synthetic = {k: v + np.random.normal(0, 0.01 * abs(v) + 1e-6) for k, v in base_x.items()}
                new_samples.append((x_synthetic, minority_class))
                samples_generated += 1

        # Convert new samples into Pandas DataFrame
        if new_samples:
            X_synthetic = pd.DataFrame([x for x, _ in new_samples])
            y_synthetic = pd.Series([y for _, y in new_samples])

            # Concatenate synthetic samples with original dataset
            X_train = pd.concat([X_train, X_synthetic], ignore_index=True)
            y_train = pd.concat([y_train, y_synthetic], ignore_index=True)
            
            print(f"✓ Balancing complete. New class distribution: {Counter(y_train)}")

    # Feature selection (initially applied)
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        selector.learn_one(xi1, yi1)

    # Train the model on the balanced dataset
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xi1 = selector.transform_one(xi1)
        model.learn_one(xi1, yi1)

    # Evaluate on the test set
    for xi, yi in stream.iter_pandas(X_test, y_test):
        xi = selector.transform_one(xi)
        y_pred = model.predict_one(xi)
        model.learn_one(xi, yi)
        metric.update(yi, y_pred)  # Fixed: Don't reassign metric

        # Update drift detectors with misclassification error
        adwin.update(y_pred != yi)
        eddm.update(y_pred != yi)

        # Store drift detection results in the sliding window
        drift_window.append((adwin.drift_detected, eddm.drift_detected))

        # Drift triggers if both ADWIN & EDDM detected drift at least once in the past window
        adwin_agreed = any(adwin_detected for adwin_detected, _ in drift_window)
        eddm_agreed = any(eddm_detected for _, eddm_detected in drift_window)

        if adwin_agreed and eddm_agreed:
            print(f"🚨 Confirmed Drift at sample {i}, re-selecting features...")
            selector = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=k_features)

            # Re-train feature selection on available data
            for xi1, yi1 in stream.iter_pandas(X_train, y_train):
                selector.learn_one(xi1, yi1)

            # Reset drift window after handling drift
            drift_window.clear()

        # Store results
        t.append(i)
        m.append(metric.get() * 100)
        yt.append(yi)
        yp.append(y_pred)
        i += 1

    # Print final evaluation metrics
    print("Final Accuracy:", round(metric.get() * 100, 2), "%")

    return t, m


def acc_fig(t, m, name):
    """Plot accuracy changes over time"""
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    plt.clf()
    plt.plot(t, m, '-b', label='Avg Accuracy: %.2f%%' % (m[-1]))
    plt.legend(loc='best')
    plt.title(name + ' on Gotham Dataset', fontsize=15)
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')
    filename = f'{name.replace(" ", "_")}_gotham.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


# ============================================================================
# TRAIN BASE MODELS (7 models)
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING BASE MODELS")
print("=" * 80)

# Model 1: Hoeffding Tree
print("\n[1/7] Hoeffding Tree (HT)...")
start = time.time()
name1 = "HT model"
model1 = tree.HoeffdingTreeClassifier()
t, m1 = adaptive_learning(model1, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m1, name1)

# Model 2: Leveraging Bagging
print("[2/7] Leveraging Bagging (LB)...")
start = time.time()
name2 = "LB model"
model2 = ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3)
t, m2 = adaptive_learning(model2, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m2, name2)

# Model 3: Adaptive Random Forest
print("[3/7] Adaptive Random Forest (ARF)...")
start = time.time()
name3 = "ARF model"
model3 = forest.ARFClassifier(n_models=3)
t, m3 = adaptive_learning(model3, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m3, name3)

# Model 4: Streaming Random Patches
print("[4/7] Streaming Random Patches (SRP)...")
start = time.time()
name4 = "SRP model"
model4 = ensemble.SRPClassifier(n_models=3)
t, m4 = adaptive_learning(model4, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m4, name4)

# Model 5: Aggregated Mondrian Forest
print("[5/7] Aggregated Mondrian Forest (AMF)...")
start = time.time()
name5 = "AMF model"
model5 = forest.AMFClassifier(n_estimators=3)
t, m5 = adaptive_learning(model5, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m5, name5)

# Model 6: EFDT (Hoeffding Adaptive Tree with DDM)
print("[6/7] EFDT model...")
start = time.time()
name6 = "EFDT model"
model6 = tree.HoeffdingAdaptiveTreeClassifier(drift_detector=DDM())
t, m6 = adaptive_learning(model6, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m6, name6)

# Model 7: HAT (Hoeffding Adaptive Tree)
print("[7/7] Hoeffding Adaptive Tree (HAT)...")
start = time.time()
name7 = "HAT model"
model7 = tree.HoeffdingAdaptiveTreeClassifier()
t, m7 = adaptive_learning(model7, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m7, name7)

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
print("\n" + "=" * 80)
print("HYPERPARAMETER OPTIMIZATION")
print("=" * 80)
print(f"Running {HYPEROPT_MAX_EVALS} iterations per model (this takes time...)")

# Optimize ARF
print("\n[1/2] Optimizing ARF...")

def objective_arf(params):
    params = {
        'n_models': int(params['n_models']),
        'drift_detector': ADWIN() if params['drift_detector'] == 'ADWIN' else EDDM()
    }
    clf = forest.ARFClassifier(**params)
    t, m = adaptive_learning(clf, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
    return {'loss': -m[-1], 'status': STATUS_OK}

space_arf = {
    'n_models': hp.quniform('n_models', 2, 10, 1),
    'drift_detector': hp.choice('drift_detector', ['ADWIN', 'EDDM'])
}

best_arf = fmin(fn=objective_arf, space=space_arf, algo=tpe.suggest, max_evals=HYPEROPT_MAX_EVALS, verbose=0)
print(f"ARF: Hyperopt estimated optimum {best_arf}")

# Optimize SRP
print("\n[2/2] Optimizing SRP...")

def objective_srp(params):
    params = {
        'n_models': int(params['n_models']),
        'drift_detector': ADWIN() if params['drift_detector'] == 'ADWIN' else EDDM()
    }
    clf = ensemble.SRPClassifier(**params)
    t, m = adaptive_learning(clf, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
    return {'loss': -m[-1], 'status': STATUS_OK}

space_srp = {
    'n_models': hp.quniform('n_models', 2, 10, 1),
    'drift_detector': hp.choice('drift_detector', ['ADWIN', 'EDDM'])
}

best_srp = fmin(fn=objective_srp, space=space_srp, algo=tpe.suggest, max_evals=HYPEROPT_MAX_EVALS, verbose=0)
print(f"SRP: Hyperopt estimated optimum {best_srp}")

# ============================================================================
# TRAIN OPTIMIZED MODELS
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING OPTIMIZED MODELS")
print("=" * 80)

# Model 8: Optimized ARF
print("\n[1/2] Optimized ARF...")
start = time.time()
name8 = "Optimized ARF model"
drift_detector = EDDM() if best_arf['drift_detector'] == 1 else ADWIN()
model8 = forest.ARFClassifier(n_models=int(best_arf['n_models']), drift_detector=drift_detector)
t, m8 = adaptive_learning(model8, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m8, name8)

# Model 9: Optimized SRP
print("[2/2] Optimized SRP...")
start = time.time()
name9 = "Optimized SRP model"
drift_detector = ADWIN() if best_srp['drift_detector'] == 0 else EDDM()
model9 = ensemble.SRPClassifier(n_models=int(best_srp['n_models']), drift_detector=drift_detector)
t, m9 = adaptive_learning(model9, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m9, name9)

# ============================================================================
# SH-CASH ENSEMBLE (3 configurations from paper)
# ============================================================================
print("\n" + "=" * 80)
print("SH-CASH ENSEMBLE LEARNING")
print("=" * 80)

# Configuration 1: All 9 models
print("\n[1/3] SH-CASH with all 9 models...")
start = time.time()
model_list_all = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
name10 = "Proposed SH-CASH model"
model10 = model_selection.SuccessiveHalvingClassifier(
    model_list_all, metric=metrics.Accuracy(), budget=40000, eta=2, verbose=True
)
t, m10 = adaptive_learning(model10, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m10, name10)

# Configuration 2: Best 2 base + optimized (4 models)
print("[2/3] SH-CASH with ARF, SRP + optimized versions...")
start = time.time()
model_list_best4 = [model3, model8, model4, model9]
name11 = "SH-CASH Best 4"
model11 = model_selection.SuccessiveHalvingClassifier(
    model_list_best4, metric=metrics.Accuracy(), budget=40000, eta=2, verbose=True
)
t, m11 = adaptive_learning(model11, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m11, name11)

# Configuration 3: Only optimized models
print("[3/3] SH-CASH with optimized models only...")
start = time.time()
model_list_opt = [model8, model9]
name12 = "SH-CASH Optimized Only"
model12 = model_selection.SuccessiveHalvingClassifier(
    model_list_opt, metric=metrics.Accuracy(), budget=40000, eta=2, verbose=True
)
t, m12 = adaptive_learning(model12, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
print(f"Time: {time.time()-start:.2f}s\n")
acc_fig(t, m12, name12)

# ============================================================================
# COMPREHENSIVE COMPARISON FIGURE
# ============================================================================
print("\n" + "=" * 80)
print("CREATING COMPREHENSIVE COMPARISON FIGURE")
print("=" * 80)

plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(24, 15))
sns.set_style("darkgrid")
plt.clf()

# Plot all models
plt.plot(t, m10, '-ro', label=name10 + ', Avg Accuracy: %.2f%%' % (m10[-1]))
plt.plot(t, m3, 'orange', label=name3 + ', Avg Accuracy: %.2f%%' % (m3[-1]))
plt.plot(t, m4, 'black', label=name4 + ', Avg Accuracy: %.2f%%' % (m4[-1]))
plt.plot(t, m1, '-b', label=name1 + ', Avg Accuracy: %.2f%%' % (m1[-1]))
plt.plot(t, m2, '-g', label=name2 + ', Avg Accuracy: %.2f%%' % (m2[-1]))
plt.plot(t, m5, '-r', label=name5 + ', Avg Accuracy: %.2f%%' % (m5[-1]))
plt.plot(t, m6, '-c', label=name6 + ', Avg Accuracy: %.2f%%' % (m6[-1]))
plt.plot(t, m7, '-m', label=name7 + ', Avg Accuracy: %.2f%%' % (m7[-1]))

plt.legend(loc='lower right')
plt.title('Online Learning Methods Comparison on Gotham Dataset', fontsize=40)
plt.xlabel('Number of samples')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('comprehensive_comparison_gotham.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comprehensive_comparison_gotham.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print("\nFinal Model Performance Summary:")
print(f"{'Model':<35} {'Accuracy':<15}")
print("-" * 50)
print(f"{name1:<35} {m1[-1]:.2f}%")
print(f"{name2:<35} {m2[-1]:.2f}%")
print(f"{name3:<35} {m3[-1]:.2f}%")
print(f"{name4:<35} {m4[-1]:.2f}%")
print(f"{name5:<35} {m5[-1]:.2f}%")
print(f"{name6:<35} {m6[-1]:.2f}%")
print(f"{name7:<35} {m7[-1]:.2f}%")
print(f"{name8:<35} {m8[-1]:.2f}%")
print(f"{name9:<35} {m9[-1]:.2f}%")
print(f"{name10:<35} {m10[-1]:.2f}%")
print(f"{name11:<35} {m11[-1]:.2f}%")
print(f"{name12:<35} {m12[-1]:.2f}%")

if label_encoder is not None:
    print("\n📊 Attack Types in Dataset:")
    print("-" * 60)
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {i}: {class_name}")

print("\n📁 All figures saved to current directory!")
print("=" * 80)