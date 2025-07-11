#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the YAML file path
YAML_FILE="$SCRIPT_DIR/config.yaml"
TMP_FILE="$SCRIPT_DIR/tmp.yaml"

cp "$YAML_FILE" "$TMP_FILE"

# Function to download a file
download_file() {
    local url=$1
    local output_path=$2
    wget -O "$output_path" "$url"
}

# Function to download and extract a ZIP file
download_and_extract_zip() {
    local url=$1
    local output_zip=$2
    local output_dir=$3

    # Download the ZIP file
    download_file "$url" "$output_zip"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Extract the ZIP file
    unzip -o "$output_zip" -d "$output_dir"

    # Remove the ZIP file after extraction
    rm "$output_zip"
}

# Function to prompt for download
prompt_download_file() {
    local url=$1
    local output_path=$2
    local description=$3

    read -p "Do you want to download the $description? (y/n): " answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        download_file "$url" "$output_path"
        return 0  # Success
    else
        return 1  # Failure
    fi
}


# Function to prompt for download of ZIP files
prompt_download_zip() {
    local url=$1
    local output_zip=$2
    local output_dir=$3
    local description=$4

    read -p "Do you want to download and extract $description? (y/n): " answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        download_and_extract_zip "$url" "$output_zip" "$output_dir"
        return 0
    else
        return 1
    fi
}

# Create directories if they don't exist
mkdir -p "$SCRIPT_DIR/cyberbattle/models/classifier"
mkdir -p "$SCRIPT_DIR/cyberbattle/data/scrape_samples/"
mkdir -p "$SCRIPT_DIR/cyberbattle/data/env_samples/"

# Read URLs from the YAML file and download each file if needed, and place in the right directory

# Classifier
ZIP_FILE_0_URL="$(grep 'vulnerability_classifier_url:' "$YAML_FILE" | cut -d ' ' -f 2- | sed 's/^"\(.*\)"$/\1/')"
ZIP_FILE_0_OUTPUT="$SCRIPT_DIR/cyberbattle/models/classifier/download.zip"
ZIP_FILE_0_EXTRACT_DIR="$SCRIPT_DIR/cyberbattle/models/classifier/"

if prompt_download_zip "$ZIP_FILE_0_URL" "$ZIP_FILE_0_OUTPUT" "$ZIP_FILE_0_EXTRACT_DIR" "default classifier"; then
    sed -i 's|vulnerability_classifier_path:.*|vulnerability_classifier_path: "classifier"|' "$TMP_FILE"
fi

# Environment database
ZIP_FILE_1_URL="$(grep 'default_database_url:' "$YAML_FILE" | cut -d ' ' -f 2- | sed 's/^"\(.*\)"$/\1/')"
ZIP_FILE_1_OUTPUT="$SCRIPT_DIR/cyberbattle/data/scrape_samples/data.zip"
ZIP_FILE_1_EXTRACT_DIR="$SCRIPT_DIR/cyberbattle/data/scrape_samples"

if prompt_download_zip "$ZIP_FILE_1_URL" "$ZIP_FILE_1_OUTPUT" "$ZIP_FILE_1_EXTRACT_DIR" "default database of services and vulnerabilities"; then
    sed -i 's|nvd_data_path:.*|nvd_data_path: "default_data"|' "$TMP_FILE"
fi

# GAE model
ZIP_FILE_2_URL="$(grep 'gae_url:' "$YAML_FILE" | cut -d ' ' -f 2- | sed 's/^"\(.*\)"$/\1/')"
ZIP_FILE_2_OUTPUT="$SCRIPT_DIR/cyberbattle/gae/logs/model.zip"
ZIP_FILE_2_EXTRACT_DIR="$SCRIPT_DIR/cyberbattle/gae/logs"

if prompt_download_zip "$ZIP_FILE_2_URL" "$ZIP_FILE_2_OUTPUT" "$ZIP_FILE_2_EXTRACT_DIR" "gae model"; then
    sed -i 's|gae_path:.*|gae_path: "GAE_2024-11-20_08-59-57"|' "$TMP_FILE"
fi

# Default scenarios
ZIP_FILE_3_URL="$(grep 'default_environments_url:' "$YAML_FILE" | cut -d ' ' -f 2- | sed 's/^"\(.*\)"$/\1/')"
ZIP_FILE_3_OUTPUT="$SCRIPT_DIR/cyberbattle/data/env_samples/graphs.zip"
ZIP_FILE_3_EXTRACT_DIR="$SCRIPT_DIR/cyberbattle/data/env_samples"

if prompt_download_zip "$ZIP_FILE_3_URL" "$ZIP_FILE_3_OUTPUT" "$ZIP_FILE_3_EXTRACT_DIR" "default scenarios"; then
    sed -i 's|default_environments_path:.*|default_environments_path: "syntethic_deployment_20_graphs_100_nodes"|' "$TMP_FILE"
fi

# Move the updated tmp file back to the original YAML file
mv "$TMP_FILE" "$YAML_FILE"

echo "Files downloaded and YAML updated successfully!"
