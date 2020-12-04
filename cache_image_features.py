import logging

from allennlp.data import DataLoader, DatasetReader
from allennlp.common.params import Params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache_image_features.py")

CONFIG = "./vilbert_vqa_from_huggingface.jsonnet"

logger.info("Reading params")
params = Params.from_file(CONFIG)

logger.info("Instantiating validation dataset reader and data loader")
validation_reader = DatasetReader.from_params(params["validation_dataset_reader"])
validation_data_loader = DataLoader.from_params(
    params["data_loader"].duplicate(),
    reader=validation_reader,
    data_path=params["validation_data_path"],
)

for instance in validation_data_loader.iter_instances():
    pass

del validation_data_loader

logger.info("Instantiating train dataset reader and data loader")
train_reader = DatasetReader.from_params(params["dataset_reader"])
data_loader = DataLoader.from_params(
    params["data_loader"].duplicate(),
    reader=train_reader,
    data_path=params["train_data_path"],
)

for instance in data_loader.iter_instances():
    pass

logger.info("Done")
