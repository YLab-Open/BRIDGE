import os
from huggingface_hub import hf_hub_download


def main(repo_dataset):
    # Download the dataset zip file from Hugging Face Hub
    file_path = hf_hub_download(
        repo_id=repo_dataset, filename="Dataset.zip", repo_type="dataset", local_dir=""
    )

    # Decompress the downloaded zip file
    unzip_command = f"unzip -q {file_path} -d dataset_raw"
    os.system(unzip_command)

    # Download the example zip file from Hugging Face Hub
    file_path = hf_hub_download(
        repo_id=repo_dataset, filename="Example.zip", repo_type="dataset", local_dir=""
    )

    # Decompress the downloaded zip file
    unzip_command = f"unzip -q {file_path} -d dataset_raw/example"
    os.system(unzip_command)


if __name__ == "__main__":
    repo_dataset = "YLab-Open/BRIDGE-Open"
    main(repo_dataset)
    print("Done.")
