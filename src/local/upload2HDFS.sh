#!/bin/bash
hadoop fs -mkdir fault_diagnosis
hadoop fs -mkdir fault_diagnosis/dataset
hadoop fs -put ../../dataset/*.csv fault_diagnosis/dataset 