python -c 'from util import preprocess; preprocess()'

python -c 'from util import decode_the_classes; decode_the_classes()'

python -c 'from util import encode_the_classes; encode_the_classes()'

kubectl cp machinelearning4/machinelearning4-group1-pod:/workspace/Vit-PestDetection/test_results.xlsx test_results.xlsx

Mean:  [0.42259824591596673, 0.5451352098593581, 0.5488909787664987]
Std:  [0.3013541560128329, 0.2690995318234087, 0.2776641671736981]