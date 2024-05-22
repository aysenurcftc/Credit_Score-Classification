
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



def train_model(X_train, y_train):
    """Train a classification model."""
    
    if config.model_type == 'random_forest':
        model = RandomForestClassifier(random_state=config.RANDOM_SEED)
    elif config.model_type == 'svm':
        model = SVC(random_state=config.RANDOM_SEED)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

