export T5_MODEL="t5-11b"
export MODEL_TEMPLATE="./templates/siamese-t5-11b-template"
rm -R $MODEL_TEMPLATE
python3 -c "from siamese_model import T5Siamese;T5Siamese.init_from_base_t5_model(\"${T5_MODEL}\", \"${MODEL_TEMPLATE}\");"
# google/t5-v1_1-xxl