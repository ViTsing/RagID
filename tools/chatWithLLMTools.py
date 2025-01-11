#llms call function
import time,os
# from openai import AzureOpenAI
# from zhipuai import ZhipuAI
from openai import OpenAI

def chatWithGpt3Point5(chat_content):
    '''
    param: 
        chat_content: input to the model
    function:
        use chatgpt 2024-02-15-preview version to chat
    '''
    api_key = os.getenv("API_KEY")

    
    # client = AzureOpenAI(
    #   azure_endpoint = "https://gptnewinstance.openai.azure.com/", 
    #   api_key= api_key,  
    #   api_version="2024-02-15-preview"
    # )
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,

    )


    max_retries=3
    # try max_retries times until reply no error or reach times limit
    for retry_count in range(max_retries):
        completion = None
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-3.5-turbo", # model = "deployment_name"
                messages = chat_content,
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            replyContent = completion.choices[0].message.content
            break
        except Exception as e:
            print(f"Retry {retry_count + 1} failed. Error: {str(e)}")
            print(completion)
            replyContent = ""
            time.sleep(1)  # Wait for 1 seconds before retrying
    return replyContent


def chatWithGLM4(chat_content):
    '''
    param:
        chat_content: input to the model
    function:
        Chatting with glm4 April 2024
    '''
    max_retries=3
    api_key = os.getenv("GLM4_API_KEY")
    # try max_retries times until reply no error or reach times limit
    for retry_count in range(max_retries):
        response = None
        try:     
            client = ZhipuAI(api_key=api_key) 
            response = client.chat.completions.create(
                model="glm-4",  
                messages= chat_content
            )
            replyContent = response.choices[0].message.content
            break
        except Exception as e:
            print(f"Retry {retry_count + 1} failed. Error: {str(e)}")
            if "400" in str(e):
                replyContent = "unknown error"
                break
            elif "429" in str(e):
                replyContent = "balance exhausted"
            time.sleep(1)  # Wait for 10 seconds before retrying
    return replyContent


    

