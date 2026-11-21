# Final Model Selection

## Selected Model

I selected XGBoost with the Without Trends feature configuration for the final evaluation. This configuration uses 22 original features from the four preceding recorded earnings events, excluding `eps_change1_2`, `eps_trend4` and `surprise_trend4`.  My decision to do this was based on the chronological development results from 2020 through 2023. XGBoost achieved the highest mean macro F1 among the original classifiers. The ablation experiments showed that removing the engineered trend features had almost no effect on predictive performance so I selected the smaller feature set. I will use the original XGBoost hyperparameters without additional tuning.

## Final Evaluation

I will train the selected model on all 82,353 pre-2024 development observations and evaluate it on the 7,996 reserved observations from 2024. I will compare its performance against the Previous Outcome and Majority Class baselines. Macro F1 will remain the primary evaluation metric. I will also report accuracy, per-class precision and recall, F1 scores, ROC-AUC and a confusion matrix.

Preprocessing will be fitted using the development data. I will use the model's existing classification threshold of 0.5 and will not adjust the model after examining the holdout results. The evaluation remains retrospective because the source records do not establish the exact historical availability of every earnings value.

I will keep the original XGBoost hyperparameters fixed for the final evaluation. The model uses 200 trees, a maximum depth of 4 and a learning rate of 0.05. I will retain the existing regularization and sampling settings with a random state of 42.

I will use the default classification threshold of 0.5. After fitting the model on the pre-2024 development observations, I will evaluate it once on the reserved 2024 data without adjusting the configuration based on those results.