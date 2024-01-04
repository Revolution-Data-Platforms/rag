import requests
import json
from uuid import uuid4
import base64
# In order to access the API endpoint, you will first need to create an Okta OIDC application of type Web Application.
# The application will need a grant type of client_credentials.
# The application will also need a custom scope created, e.g., "api".
# From the application, you will need to get the client_id and client_secret.

okta_domain = "ciena.okta.com"
client_id = "0oa1tm64vaai5tKVh0h8"
client_secret = "SmDr1iZ2p09OWETcCGMWX5TtHSZjxLzUgisqGxBAx33SLCEcQXmpMMCSG20AvwT4"
okta_custom_scope = "api"

system_message = "When I ask you a question you will reply with at least one joke"

# Base address of dev ciena.gpt api. Use https://localhost:44396/ for local dev
base_address = "https://gptapidev.cs.ciena.com/"

def get_okta_token():
    token_endpoint = f"https://{okta_domain}/oauth2/default/v1/token"
    
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()}"
    }
    
    data = {
        "grant_type": "client_credentials",
        "scope": okta_custom_scope
    }
    
    response = requests.post(token_endpoint, headers=headers, data=data)
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json["access_token"]
    else:
        raise Exception(f"Failed to retrieve token. Status code: {response.status_code}")

def get_user_id(jwt_token):
    headers = {
        "Authorization": f"Bearer {jwt_token}"
    }
    
    response = requests.get(f"{base_address}api/Auth/GetUserInfo", headers=headers)
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json["result"]["id"]
    else:
        raise Exception(f"Request failed with status code: {response.status_code}")

def create_completion(jwt_token, user_id, conversation_identifier, message):
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "userId": str(user_id)
    }
    
    json_payload = {
        "conversationIdentifier": conversation_identifier,
        "choicesPerPrompt": 1,
        "maxTokens": 350,
        "systemMessage": system_message,
        "message": {
            "content": message,
            "role": "user",
        },
        "nucleusSamplingFactor": 0,
        "presencePenalty": 0,
        "temperature": 0.5
    }
    
    endpoint_url = f"{base_address}api/openai/createcompletion"
    
    response = requests.post(endpoint_url, headers=headers, json=json_payload)
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json["result"]["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Request failed with status code: {response.status_code}")

def main():
    jwt_token = get_okta_token()
    user_id = get_user_id(jwt_token)
    conversation_identifier = str(uuid4())

    question1 = "Q: Who is Ciena corp?"
    question2 = "Q: Who is their CEO?"
    question3 = "Q: What is their flagship product?"

    print(question1)
    response1 = create_completion(jwt_token, user_id, conversation_identifier, question1)
    print(f"A: {response1}\n")

    # print(question2)
    # response2 = create_completion(jwt_token, user_id, conversation_identifier, question2)
    # print(f"A: {response2}\n")

    # print(question3)
    # response3 = create_completion(jwt_token, user_id, conversation_identifier, question3)
    # print(f"A: {response3}\n")

if __name__ == "__main__":
    main()
