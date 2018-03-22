#!/bin/bash
hadoop fs -mkdir dataset
hadoop fs -put ../../dataset/*.csv dataset 