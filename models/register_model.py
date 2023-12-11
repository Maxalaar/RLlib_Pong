from ray.rllib.models import ModelCatalog
from models.cnn import CustomCNN

ModelCatalog.register_custom_model('custom_cnn', CustomCNN)
