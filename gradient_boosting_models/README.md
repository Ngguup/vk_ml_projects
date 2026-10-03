# Реализация Gradient Boosting

Проверка в: evaluation.ipynb

Самостоятельно реализовал Gradient Boosting для задач регрессии и бинарной классификации и сравнил его с готовыми реализациями GradientBoostingRegressor и GradientBoostingClassifier из scikit-learn на датасетах Diabetes и Banknote Authentication.

В качестве базовых моделей использовал DecisionTreeRegressor. Реализовал последовательное обучение деревьев на ошибках текущего ансамбля и регулирование их вклада с помощью learning_rate.

Реализовал:
- GBCustomRegressor — Gradient Boosting для регрессии
- GBCustomClassifier — Gradient Boosting для бинарной классификации с использованием sigmoid

Стек: Python, NumPy, scikit-learn, Decision Trees, Gradient Boosting