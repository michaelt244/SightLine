import google.generativeai as genai

genai.configure(api_key="AIzaSyB9aYfBypH8gBUJn7RxF-1D3gBIjh1C-1I")
for model in genai.list_models():
    print(model.name)