# vilbert_vqa

## Submitting an experiment on beaker

1. Edit the training config `./vilbert_vqa_from_huggingface.jsonnet` or create a new one.

2. Commit and push your changes to the training config.

3. Edit the beaker config `./beaker.yml` to point to the commit of the training config.

4. Submit the beaker experiment:

    ```bash
    beaker experiment create \
        --workspace ai2/ViLBERT-VQA \  # change as needed
        --file beaker.yml \
        --name try-030  # change as needed
    ```
