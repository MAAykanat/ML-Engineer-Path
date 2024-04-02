import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus, graphviz

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, cross_validate, validation_curve
from sklearn.model_selection import GridSearchCV

from skompiler import skompile

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df.head())

#################
### MODELLING ###
#################

####################
## Holdout Method ##
####################

# 1. Splitting the data
y = df['Outcome']
X = df.drop('Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cart_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# 2. Prediction
# for Confusion Matrix y_pred
y_pred = cart_model.predict(X_test)
# for AUC y_prob
y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test,y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test, y_prob))

######################
## Cross Validation ##
######################

cv_result = cross_validate(estimator=cart_model,
                           X=X_train,
                           y=y_train,
                           cv=10,
                           scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

print(cv_result["test_accuracy"].mean())
print(cv_result["test_roc_auc"].mean())

#################################
## Hyperparameter Optimization ##
#################################

# Grid Search

print(cart_model.get_params())

cart_params = {"max_depth": range(1,15),
                "min_samples_split": list(range(2,50))}

cart_model_best_grid = GridSearchCV(estimator=cart_model,
                                    param_grid=cart_params,
                                    cv=10,
                                    n_jobs=-1,
                                    verbose=1).fit(X_train, y_train)
print("#########RESULT MODEL#########")
print(cart_model_best_grid.best_params_)
print(cart_model_best_grid.best_score_)

#################
## Final Model ##
#################

cart_final = cart_model.set_params(**cart_model_best_grid.best_params_).fit(X_train, y_train)

# for Confusion Matrix y_pred
y_pred = cart_model.predict(X_test)
# for AUC y_prob
y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test,y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test, y_prob))

###########################
## 6. Feature Importance ##
###########################

cart_final.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    if save:
        plt.savefig('importances.png')

    plt.show()

# plot_importance(model=cart_final, features=X)

########################################################
## 7. Analyzing Model Complexity with Learning Curves ##
########################################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10, save = False):
    """
    Function to plot validation curve for a given model

    Parameters
    ----------
    model : object
        Model to be used
    X : array-like
        Feature dataset
    y : array-like
        Target dataset
    param_name : str
        Name of the parameter to be optimized
    param_range : array-like
        Range of the parameter to be optimized
    scoring : str
        Scoring metric 
    cv : int
        Number of cross-validation folds
    save : bool
        Save the plot as a png file

    Returns
    -------
    None
    """
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    if save:
        plt.savefig(f"validation_curve_{type(model).__name__}_{param_name}.png")
    plt.show(block=True)

"""
val_curve_params(cart_final, X_train, y_train, "max_depth", range(1, 15),save=False)

cart_val_params = [["max_depth", range(1, 15)], ["min_samples_split", range(2, 50)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])
"""

#######################
## 8. Visualization  ##
#######################
    
"""
DO IT LATER

"""

################################################
# 10. Extracting Python Codes of Decision Rules
################################################

# expr = skompile(cart_final.predict)
# sql = expr.to('sqlalchemy/sqlite')

# print(sql)

# print(skompile(cart_final.predict).expr.to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))