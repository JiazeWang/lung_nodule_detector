sid="./preprocess/hku_list_test.csv"
submit="/research/dept8/jzwang/code/lung_nodule_detector/hku_submission_6912.csv"


mkdir ./evaluationScript_py3/exampleFiles/hku_submission_6912
python ./evaluationScript_py3/noduleCADEvaluationLUNA16.py \
        ./preprocess/annotations_hku.csv \
        ./evaluationScript_py3/annotations/annotations_excluded.csv \
        ${sid} \
        ${submit} \
        ./evaluationScript_py3/exampleFiles/hku_submission_6912/
