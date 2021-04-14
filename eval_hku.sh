sid="./preprocess/hku_list_test.csv"
submit="'./hku_submission_5.csv'"



mkdir ./evaluationScript_py3/exampleFiles/res18_split_30
python ./evaluationScript_py3/noduleCADEvaluationLUNA16.py \
        ./preprocess/annotations_hku.csv \
        ./evaluationScript_py3/annotations/annotations_excluded.csv \
        ${sid} \
        ${submit} \
        ./evaluationScript_py3/exampleFiles/res18_split_30/
