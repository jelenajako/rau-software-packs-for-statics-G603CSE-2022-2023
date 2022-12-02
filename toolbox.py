import numpy as np


def split_data(df):
    clean_data = df.drop(["Formatted Date", "Summary", "Loud Cover", "Daily Summary"], axis=1)
    y = clean_data["Precip Type"]
    X = clean_data.drop("Precip Type", axis=1)
    return X, y


def remove_df_outliers_iqr(df):
    columns = df.columns
    for column in columns:
        if df[column].dtype != object:
            X = np.array(df[column].values)
            LQ = np.quantile(X, 0.25)
            UQ = np.quantile(X, 0.75)
            IQR = UQ - LQ
            lower_limit = LQ - 1.5 * IQR
            upper_limit = UQ + 1.5 * IQR

            df[df[column] < lower_limit] = None
            df[df[column] > upper_limit] = None
            df = df.dropna()

    return df

def correlation(x1, x2):
    # sum( (x1i - x10) * (x2i - x20) ) / sqrt (sum (x1i - x10)^2 * sum (x2i - x20) ^ 2)
    x10 = np.mean(x1)
    x20 = np.mean(x2)

    x1_dif = x1 - x10
    x2_dif = x2 - x20

    x1_dif_sq = x1_dif ** 2
    x2_dif_sq = x2_dif ** 2

    top = np.sum(x1_dif * x2_dif)
    bottom = np.sqrt(np.sum(x1_dif_sq) * np.sum(x2_dif_sq))

    try:
        coef = top / bottom
        return coef
    except Exception as e:
        raise e


def compute_correlation_matrix(X):
    correlation_matrix = []

    for column1 in X.columns:
        correlation_row = []
        for column2 in X.columns:
            x1 = np.array(X[column1].values)
            x2 = np.array(X[column2].values)
            coeff = correlation(x1, x2)
            correlation_row.append(coeff)

        correlation_matrix.append(correlation_row)

    correlation_matrix = np.array(correlation_matrix)
    return correlation_matrix


def remove_correlated_columns(correlation_matrix, X):
    min_correlation_coeff_threshold = -0.75
    max_correlation_coeff_threshold = 0.75

    columns_to_remove = []
    n_features = len(X.columns)
    for i in range(n_features):
      for j in range(i+1, n_features):
        if correlation_matrix[i, j] > max_correlation_coeff_threshold or correlation_matrix[i, j] < min_correlation_coeff_threshold:
          columns_to_remove.append(j)

    columns_to_remove_names = []
    for col_index in columns_to_remove:
      columns_to_remove_names.append(X.columns[col_index])
    X = X.drop(columns_to_remove_names, axis=1)
    return X


def min_max_scale(x):
    x_min = np.min(x)
    x_max = np.max(x)

    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled

