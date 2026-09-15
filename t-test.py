from scipy import stats

stratified_kfold_scores = [1.0000, 0.9973, 0.9978, 0.9991, 0.9987]
random_split_scores = [0.9982, 0.9987, 0.9987, 1.0000, 0.9991]  # seed order: 10,20,30,40,42

t_stat, p_value = stats.ttest_ind(random_split_scores, stratified_kfold_scores, equal_var=False)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Result: Performance difference is NOT statistically significant (p > 0.05)")
else:
    print("Result: Performance difference IS statistically significant (p <= 0.05)")