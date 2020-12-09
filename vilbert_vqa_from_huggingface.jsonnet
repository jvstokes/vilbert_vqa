local model_name = "bert-base-uncased";
local effective_batch_size = 256;
local gpu_batch_size = 64;
local num_gpus = 1;
local seed = 2;
local learning_rate = 4e-5;
local num_workers = 1;
// local data_dir = std.extVar("HOME") + "/data/vqa";
local data_dir = "/data/vqa";

local construct_vocab = false;

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      "directory": "https://storage.googleapis.com/allennlp-public-data/vqav2/vilbert_vqa_balanced_real.vocab.tar.gz"
    };

{
  "dataset_reader": {
    "type": "vqav2",
    "image_dir": data_dir + "/images",
    "feature_cache_dir": data_dir + "/feature_cache",
    "feature_cache_read_only": true,
    "image_loader": "torch",
    // "image_featurizer": "resnet_backbone",
    // "region_detector": "faster_rcnn",
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
    // "max_instances": 1000,
    "image_processing_batch_size": 16,
    "answer_vocab": if construct_vocab then null else vocabulary,
    "run_image_feature_extraction": !construct_vocab,
    "multiple_answers_per_question": !construct_vocab
  },
  "validation_dataset_reader": self.dataset_reader {
    "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
  },
  "vocabulary": vocabulary,
  "train_data_path": ["balanced_real_train", "balanced_real_val[1000:]"],
  "validation_data_path": "balanced_real_val[:1000]",
  "model": {
    "type": "vqa_vilbert_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 1024,
    "image_hidden_size": 1024,
    "image_num_attention_heads": 8,
    "image_num_hidden_layers": 6,
    "combined_hidden_size": 1024,
    "combined_num_attention_heads": 8,
    "pooled_output_dim": 1024,
    "image_intermediate_size": 1024,
    "image_attention_dropout": 0.1,
    "image_hidden_dropout": 0.1,
    "image_biattention_id": [0, 1, 2, 3, 4, 5],
    "text_biattention_id": [6, 7, 8, 9, 10, 11],
    "text_fixed_layer": 0,
    "image_fixed_layer": 0,
    "fusion_method": "mul"
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    "num_workers": num_workers,
    // "start_method": "spawn",
    [if !construct_vocab then "max_instances_in_memory"]: gpu_batch_size * 2
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": learning_rate,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        // text stream encoder
        [
          [
            "^embeddings\\.*",
            "^encoder\\.layers1\\..*",
          ],
          {}
        ],
        // vision stream encoder, cross-stream connections, poolers, and classifier head
        [
          [
            "^image_embeddings\\.*",
            "^encoder\\.layers2\\..*",
            "^encoder\\.c_layer\\..*",
            "^t_pooler\\..*",
            "^v_pooler\\..*",
            "^classifier\\..*",
          ],
          {}
        ],
      ],
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "gradual_unfreezing": true,
      "discriminative_fine_tuning": true,
      "cut_frac": 0.1,
      "num_steps_per_epoch": std.ceil(658111 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"]),
    },
    "validation_metric": "+fscore",
    // "patience": 5,
    "num_epochs": 30,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus),
    "tensorboard_writer": {
        "summary_interval": 10,
        "should_log_learning_rate": false,
        "should_log_parameter_statistics": false,
    },
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed,
}
