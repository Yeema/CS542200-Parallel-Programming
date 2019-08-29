#!/bin/bash

# Do not uncomment these lines to directly execute the script
# Modify the path to fit your need before using this script
input=$1

if [[ "$2" == "" ]]; then
	iteration=1
else
	iteration=$2
fi

INPUT_FILE=/user/ta/PageRank/input-$input
OUTPUT_DIR=out
OUTPUT_FILE=out/result
JAR=pageRank.jar

hdfs dfs -rm -R $OUTPUT_DIR
hadoop jar $JAR pageRank.pageRank $INPUT_FILE $OUTPUT_DIR $iteration
hdfs dfs -getmerge $OUTPUT_FILE pagerank_$input.out
/usr/local/bin/hw5-judge $input $iteration pagerank_$input.out

