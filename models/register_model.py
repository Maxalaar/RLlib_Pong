from ray.rllib.models import ModelCatalog
from models.cnn import CustomCNN
from models.decision_tree import DecisionTree

ModelCatalog.register_custom_model('custom_cnn', CustomCNN)
ModelCatalog.register_custom_model('decision_tree', DecisionTree)
