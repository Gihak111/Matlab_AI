from transformers import pipeline

def get_hugging_model(task='text-classification'):
    # Hugging Face 파이프라인 불러오기
    model = pipeline(task)
    return model
