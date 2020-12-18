local model_name = "bert-base-uncased";
local vocab_size = 30522;
local effective_batch_size = 128;
local gpu_batch_size = 128;
local num_gpus = 1;

local construct_vocab = true;

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      // CHANGE LOC
      "directory": "/Users/jacksons/Projects/gqa-train/vocabulary/vilbert_gqa_train.vocab.tar.gz"
    };

{
  "dataset_reader": {
    "type": "gqa",
    // CHANGE LOC
    "image_dir": "/Users/jacksons/Projects/gqa-train/images",
    "feature_cache_dir": "/Users/jacksons/Projects/gqa-train/feature_cache",
    "image_loader": "torch",
    "image_featurizer": "resnet_backbone",
    "region_detector": "faster_rcnn",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name
      }
    },
    // Change max instances
    "max_instances": 100,
    "image_processing_batch_size":32,
    "answer_vocab": if construct_vocab then null else vocabulary,
    "run_image_feature_extraction": !construct_vocab,
    "multiple_answers_per_question": !construct_vocab
  },
  "validation_dataset_reader": self.dataset_reader {
    "keep_unanswerable_questions": true
  },
  "vocabulary": vocabulary,
  "train_data_path": "testdev_all",
  "validation_data_path": "val_balanced",
  "model": {
    "type": "vqa_vilbert",
    "text_embeddings": {
      "vocab_size": vocab_size,
      "hidden_size": 768,
      "pad_token_id": 0,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "dropout": 0.1
    },
    "image_embeddings": {
      "feature_dim": 1024,
      "hidden_dim": 1024
    },
    "encoder": {
      # text
      "hidden_size1": 768,
      "num_hidden_layers1": 12,
      "intermediate_size1": 3072,
      "num_attention_heads1": 12,
      "attention_dropout1": 0.1,
      "hidden_dropout1": 0.1,
      "biattention_id1": [6, 7, 8, 9, 10, 11],
      "fixed_layer1": 0,

      # vision
      "hidden_size2": 1024,
      "num_hidden_layers2": 6,
      "intermediate_size2": 1024,
      "num_attention_heads2": 8,
      "attention_dropout2": 0.1,
      "hidden_dropout2": 0.1,
      "biattention_id2": [0, 1, 2, 3, 4, 5],
      "fixed_layer2": 0,

      "combined_num_attention_heads": 8,
      "combined_hidden_size": 1024,
      "activation": "gelu",
    },
    "pooled_output_dim": 1024,
    "fusion_method": "mul"
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    "max_instances_in_memory": 1024
  },
  [if num_gpus > 1 then "distributed"]: {
    #"cuda_devices": std.range(0, num_gpus - 1)
    "cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  "trainer": {
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 4e-5
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 5000,
      "num_steps_per_epoch": std.ceil(600000 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"])
    },
    "validation_metric": "+fscore",
    "patience": 5,
    "num_epochs": 20,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus)
  },
  "random_seed": 876170670,
  "numpy_seed": 876170670,
  "pytorch_seed": 876170670,
}

