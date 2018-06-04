from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from framework.base import ModelEvaluator
from framework.sklearnmodels.linear_model import LogisticRegressionClassifierGenerator

if __name__ == '__main__':
    x, y = load_iris(True)
    y = y.reshape((y.shape[0], -1))
    lr_generator = LogisticRegressionClassifierGenerator()
    lr_models = LogisticRegression()
    evaluator = ModelEvaluator(model_generator=lr_generator, train_x=x, train_y=y)
    hyper_params = ['0.000001', '0.01']  # 'tol', 'C'
    eval_result = evaluator.evaluate(hyper_params)
