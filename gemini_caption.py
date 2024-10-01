from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Part

# Load Gemini Pro Vision
gemini_pro_vision_model = GenerativeModel("gemini-1.0-pro-vision")

image = Part.from_uri("gs://wppai-research-lab/fine_tune_base_model/compare_gpt4v_gemini/test/01_PUMA_MCA_ST_Line_Front_34_Dynmic_resized.jpg", mime_type="image/jpeg")
model_response = gemini_pro_vision_model.generate_content(["You are a photography and marketing expert. I'm going to show you an image. I want you to then describe the image in detail and accurately. Provide details but only include elements if you are absolutely certain that they appear in the image. If you are not certain, do not mention them in your description.", image])

print("model_response\n",model_response)

# %%
from google.cloud import storage

bucket_name = "wppai-research-lab"

client = storage.Client()
bucket = client.get_bucket(bucket_name)

prefix = "fine_tune_base_model/compare_gpt4v_gemini/test"

blobs = list(bucket.list_blobs(prefix = prefix))

import os
uri_list = []
for blob in blobs:
    if "." in blob.name.split("/")[-1]:
         #print(blob.name)
         uri = "gs://" + bucket_name + "/" + blob.name
    uri_list.append(uri)
print(uri_list)

c = 0
txt_list = []
for uri in uri_list[:2]:
     image = Part.from_uri(uri, mime_type="image/jpeg")
     model_response = gemini_pro_vision_model.generate_content(["You are a photography and marketing expert. I'm going to show you an image. I want you to then describe the image in detail and accurately. Provide details but only include elements if you are absolutely certain that they appear in the image. If you are not certain, do not mention them in your description.", image])

     #print("model_response\n",model_response)
     #print(model_response.candidates[0].content.parts[0].text)

     txt = model_response.candidates[0].content.parts[0].text
     c+=1
     print(c)
     print(txt)
     txt_list.append(txt)

for txt in txt_list[:2]:
    model_response = gemini_pro_vision_model.generate_content(f"can you take the most prominent information from the below narations and also starting with 'a photo of' in one sentence: {txt}")
    print(model_response)