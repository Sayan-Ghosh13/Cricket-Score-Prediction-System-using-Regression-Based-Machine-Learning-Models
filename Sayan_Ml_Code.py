import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data(filepath='ipl_data.csv'):
    df = pd.read_csv(filepath)
    columns_to_drop = ['mid', 'date', 'striker', 'non-striker']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df[df['overs'] >= 5.0]

    categorical_cols = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
    top_venues = df['venue'].value_counts().nlargest(10).index
    df['venue'] = df['venue'].apply(lambda x: x if x in top_venues else 'Other_Venue')

    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_processed.drop('total', axis=1)
    y = df_processed['total']

    return X, y

def train_and_evaluate_model_test(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"--- {model_name} Testing Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("-" * (len(model_name) + 6) + "\n")

    metrics = {'MAE': mae, 'MSE': mse, 'R2': r2}
    return y_pred, metrics

def train_and_evaluate_model_val(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    
    print(f"--- {model_name} Validation Set ---")
    print(f"Mean Absolute Error MAE: {mae_val:.2f}")
    print(f"Mean Squared Error MSE: {mse_val:.2f}")
    print(f"R-squared R2: {r2_val:.4f}")
    
    metrics_val = {'MAE': mae_val, 'MSE': mse_val, 'R2': r2_val}
    
    return y_pred_val, metrics_val


def evaluate_feature_group_impact(model_class, X_train, y_train, X_val, y_val, feature_groups):
    print("Evaluating impact of dropping each feature group on Linear Regression...")
    original_model = model_class()
    original_model.fit(X_train, y_train)
    y_pred_orig = original_model.predict(X_val)
    base_r2 = r2_score(y_val, y_pred_orig)
    print(f"Base R2 with all features: {base_r2:.4f}\n")

    for group_name, cols in feature_groups.items():
        if len(cols) == 0:
            print(f"No columns found for group {group_name}, skipping.")
            continue
        X_train_mod = X_train.drop(columns=cols)
        X_val_mod = X_val.drop(columns=cols)
        model = model_class()
        model.fit(X_train_mod, y_train)
        y_pred = model.predict(X_val_mod)
        r2 = r2_score(y_val, y_pred)
        print(f"R2 without {group_name}: {r2:.4f} (Change: {r2 - base_r2:+.4f})")
    print()

def plot_predictions(y_test, y_pred, model_name):
    x = np.array(y_test).reshape(-1, 1)
    y = np.array(y_pred)

    reg = LinearRegression()
    reg.fit(x, y)

    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.3, label='Data Points')
    plt.plot(x_line, y_line, color='red', lw=2, linestyle='--', label='Regression Line')
    plt.title(f'Actual vs. Predicted Total Runs ({model_name})')
    plt.xlabel('Actual Total Runs')
    plt.ylabel('Predicted Total Runs')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', lw=2, linestyle='--', label='Ground Truth')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    r2_scores = [m['R2'] for m in metrics_dict.values()]

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'purple', 'brown']
    bars = plt.bar(models, r2_scores, color=colors)
    plt.title('Model Comparison : R-squared (R²) Score')
    plt.ylabel('R-squared (R²)')
    plt.xlabel('Model name')
    plt.ylim(min(r2_scores) - 0.02, max(r2_scores) + 0.02)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom')
    plt.show()

def main():
    X, y = load_and_preprocess_data()
    if X is None:
        return

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=20/90, random_state=42)

    numerical_cols = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    lr_model = LinearRegression()
    lasso_model = Lasso(alpha=10.0, random_state=42)
    ridge_model = Ridge(alpha=10.0, random_state=42)
    elasticnet_model = ElasticNet(alpha=10.0, l1_ratio=0.5, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=36, random_state=12)

    models = {
        "Linear Regression": lr_model,
        "Lasso Regression": lasso_model,
        "Ridge Regression": ridge_model,
        "Elastic Net Regression": elasticnet_model,
        "Random Forest Regression": rf_model
    }

    predictions = {}
    all_metrics = {}
    all_metrics_val = {}

    print("Training models and evaluating on validation set...")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_val, metrics_val = train_and_evaluate_model_val(model, X_train, y_train, X_val, y_val, name)
        y_pred , metrics = train_and_evaluate_model_test(model,X_train, y_train, X_test, y_test, name)
        predictions[name] = y_pred_val
        all_metrics[name] = metrics
        all_metrics_val[name] = metrics_val

    feature_groups = {
        'Batsman': [col for col in X_train.columns if col.startswith('batsman_')],
        'Bowler': [col for col in X_train.columns if col.startswith('bowler_')],
        'Venue': [col for col in X_train.columns if col.startswith('venue_')],
        'Batting Team': [col for col in X_train.columns if col.startswith('bat_team_')],
        'Bowling Team': [col for col in X_train.columns if col.startswith('bowl_team_')],
        'Numerical': numerical_cols
    }

    evaluate_feature_group_impact(LinearRegression, X_train, y_train, X_val, y_val, feature_groups)

    print("Generating plots on validation set...")
    plot_predictions(y_val, predictions["Linear Regression"], "Linear Regression")
    plot_predictions(y_val, predictions["Lasso Regression"], "Lasso Regression")
    plot_predictions(y_val, predictions["Ridge Regression"], "Ridge Regression")
    plot_predictions(y_val, predictions["Elastic Net Regression"], "Elastic Net Regression")
    plot_predictions(y_val, predictions["Random Forest Regression"], "Random Forest Regression")
    plot_metrics_comparison(all_metrics_val)
    plot_metrics_comparison(all_metrics)

if __name__ == "__main__":
    main()
