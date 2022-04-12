from sklearn import svm
from sklearn.model_selection import GridSearchCV
import models


class SVMModel(models.BaseModel):
    def __init__(self, device, probability=False):
        super().__init__(device)
        self.model = svm.SVC(decision_function_shape='ovr',
                             probability=probability, max_iter=1000)

    def extract(self):
        pass

    def fit(self, X_train, y_train):
        print("Fit SVM Model...")
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]

        # Apply GridSearchCV to find best parameters
        cv = GridSearchCV(self.model, tuned_parameters, refit=True, verbose=3)
        cv.fit(X_train, y_train)

        # Display parameters selected by GridSearchCV
        print("Best parameters to apply are:", cv.best_params_)

        # Display model after hyperparameter tuning
        self.model = cv.best_estimator_
        print("Model after tuning is:\n", self.model)

    def predict(self, X_test, *args, **kwargs):
        [y_test] = self.model.predict(X_test)  # returns only one element
        return y_test
