#!/bin/bash
mkdir ~/data
mkdir ~/data/raw
cd ~/data/raw/

echo -n "ISPRS username:"
read username

echo -n "ISPRS password:"
read -s password

if [ -f "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip" ]; then
    echo "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip exists"
else
    wget ftp://$username:$password@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip
fi
mkdir ~/data/raw/gts
unzip ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip -d ~/data/raw/gts/

if [ -f "ISPRS_semantic_labeling_Vaihingen.zip" ]; then
    echo "ISPRS_semantic_labeling_Vaihingen.zip exists"
else
    wget ftp://$username:$password@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip
fi
unzip ISPRS_semantic_labeling_Vaihingen.zip
