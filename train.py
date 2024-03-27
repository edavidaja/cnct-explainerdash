# https://explainerdashboard.readthedocs.io/en/latest/index.html#a-more-extended-example

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive

X_train, y_train, X_test, y_test = titanic_survive()

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

explainer = ClassifierExplainer(
                model, X_test, y_test,
                # optional:
                cats=['Sex', 'Deck', 'Embarked'],
                labels=['Not survived', 'Survived'])

db = ExplainerDashboard(explainer, title="Titanic Explainer",
                    whatif=False, # you can switch off tabs with bools
                    shap_interaction=False,
                    decision_trees=False)

# https://explainerdashboard.readthedocs.io/en/latest/deployment.html#storing-explainer-and-running-default-dashboard-with-gunicorn
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)