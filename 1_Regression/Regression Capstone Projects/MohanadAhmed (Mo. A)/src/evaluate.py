from train import *

best_tuned_model = tuned_models[best_model_name]
# Predict on test set
Y_test_pred = best_tuned_model.predict(X_test)
# Compute performance metrics
test_r2 = r2_score(Y_test, Y_test_pred)
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
test_mae = mean_absolute_error(Y_test, Y_test_pred)

print(f"📊 Test R²: {test_r2:.4f}")
print(f"📊 Test RMSE: {test_rmse:.4f}")
print(f"📊 Test MAE: {test_mae:.4f}")

