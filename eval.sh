sid="./preprocess/hku_list_test.csv"
submit="hku_submission_6912.csv'"

#cp ${sid} ./evaluationScript/annotations/
#cp ${submit} ./evaluationScript/exampleFiles/submission/

#cd evaluationScript
mkdir ./evaluationScript_py3/exampleFiles/res18_split_6
python ./evaluationScript_py3/noduleCADEvaluationLUNA16.py \
        ./evaluationScript_py3/annotations/annotations.csv \
        ./evaluationScript_py3/annotations/annotations_excluded.csv \
        ${sid} \
        ${submit} \
        ./evaluationScript_py3/exampleFiles/res18_split_6/

mkdir ./evaluationScript_py3/exampleFiles/res18_split_30
python ./evaluationScript_py3/noduleCADEvaluationLUNA16.py \
        ./preprocess/annotations_hku.csv \
        ./evaluationScript_py3/annotations/annotations_excluded.csv \
        ${sid} \
        ${submit} \
        ./evaluationScript_py3/exampleFiles/res18_split_30/
