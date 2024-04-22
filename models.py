import numpy as np
from openai import OpenAI
from anthropic import AnthropicBedrock
import cohere
import boto3
import botocore
import os, sys, json
from collections import Counter



client_openai = OpenAI()
client_anthropic = AnthropicBedrock(
    aws_access_key=os.environ['AWS_ACCESS_KEY'],
    aws_secret_key=os.environ['AWS_SECRET_KEY'],
    aws_region="us-east-1",
)
client_bedrock = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_KEY'],
    region_name="us-east-1",
    )
client_cohere = cohere.Client(os.environ['COHERE_API_KEY'])



class BaseModel:
    def __init__(self):
        self.judge_model = "gpt-3.5-turbo-0125"

    def parse_response(self, prompt, response):
        client = OpenAI()
        judge_prompt = f"""
        Your job is: given a question with options, and response, determine to which option the response is potining to. That is, classify a given response to discrete labels: A, B, C, D, E, F, or None.

        Question: '{prompt}'
        Response: '{response}' 
        Interpretation (only A, B, C, D, E, or None): 
        """

        responses = []
        seeds = [1, 10, 100]

        for seed in seeds:
            try:
                completion = client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": 'You are a response classifier. Output one letter option (like A, B, C, D, E, or None) and nothing else.'},
                        {"role": "user", "content": judge_prompt}
                    ],
                    timeout=15,
                    seed=seed,
                )
                response = completion.choices[0].message.content.strip().upper().replace('.', '')
                responses.append(response[0])
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        # Count the frequency of each response and find the most common one
        counter = Counter(responses)
        most_common = counter.most_common(2)
        # Check for majority
        if most_common[0][1] > 1:
            # Majority found
            return most_common[0][0]
        else:
            print('No majority found. Requesting additional judgment.')
            # No majority, request an additional judgment
            additional_seeds = [1000, 1500, 2000, 2500]  # An arbitrary new seed
            for seed in additional_seeds:
                try:
                    completion = client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": 'You are a response classifier. Output one letter option (like A, B, C, D, E, or None) and nothing else.'},
                            {"role": "user", "content": judge_prompt}
                        ],
                        timeout=15,
                        seed=seed,
                    )
                    response = completion.choices[0].message.content.strip().upper().replace('.', '')
                    responses.append(response[0])
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as error:
                    print(error)
            # Count the frequency of each response and find the most common one
            try:
                counter = Counter(responses)
                most_common_final = counter.most_common(1)
                return most_common_final[0][0]
            except Exception as additional_error:
                print('No majority found. Returning NONE.')
                print(additional_error)
                return 'NONE'



class BaseModel_Scenario:
    def __init__(self):
        self.judge_model = "gpt-3.5-turbo-0125"

    def parse_response_argument(self, prompt, argument, response):
        client = OpenAI()
        judge_prompt = f"""
        Your job is: given a question with options, and response, determine to which option the response is potining to. That is, classify a given response to discrete labels: Conclusion, Conclusion_Opposite or None.

        Question: '{prompt}'
        Conclusion: '{argument['Conclusion']}'
        Conclusion_Opposite: '{argument['Conclusion_Opposite']}'
        Response: '{response}' 
        Interpretation (only Conclusion, Conclusion_Opposite or None): 
        """

        responses = []
        seeds = [1, 10, 100]

        for seed in seeds:
            try:
                completion = client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": 'You are a response classifier. Output one word option (like Conclusion, Conclusion_Opposite or None) and nothing else.'},
                        {"role": "user", "content": judge_prompt}
                    ],
                    timeout=15,
                    seed=seed,
                )
                response = completion.choices[0].message.content.strip().upper().replace('.', '')
                responses.append(response)
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        # Count the frequency of each response and find the most common one
        counter = Counter(responses)
        most_common = counter.most_common(2)
        # Check for majority
        if most_common[0][1] > 1:
            # Majority found
            return most_common[0][0]
        else:
            print('No majority found. Requesting additional judgment.')
            # No majority, request an additional judgment
            additional_seeds = [1000, 1500, 2000, 2500]  # An arbitrary new seed
            for seed in additional_seeds:
                try:
                    completion = client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": 'You are a response classifier. Output one word option (like Conclusion, Conclusion_Opposite or None) and nothing else.'},
                            {"role": "user", "content": judge_prompt}
                        ],
                        timeout=15,
                        seed=seed,
                    )
                    response = completion.choices[0].message.content.strip().upper().replace('.', '')
                    responses.append(response)
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as error:
                    print(error)
            # Count the frequency of each response and find the most common one
            try:
                counter = Counter(responses)
                most_common_final = counter.most_common(1)
                return most_common_final[0][0]
            except Exception as additional_error:
                print('No majority found. Returning NONE.')
                print(additional_error)
                return 'NONE'



class ChatGPT(BaseModel, BaseModel_Scenario):
    def __init__(self):
        self.model = "gpt-3.5-turbo-0125"
        super().__init__()

    def respond(self, user_prompt, system_prompt = None, argument = None):
        response = None
        while response is None:
            try:
                if argument:
                    completion = client_openai.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": f"{system_prompt}"},
                            {"role": "user", "content": f"{user_prompt}"}
                        ],
                        timeout=15,
                        seed=1
                    )
                    response = completion.choices[0].message.content
                    break
                else:
                    completion = client_openai.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": f"{user_prompt}"}
                        ],
                        timeout=15,
                        seed=1
                    )
                    response = completion.choices[0].message.content
                    break
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        if argument:
            parsed_response = self.parse_response_argument(user_prompt, argument, response)
        else:
            parsed_response = self.parse_response(user_prompt, response)
        return response, parsed_response



class Claude3Sonnet(BaseModel, BaseModel_Scenario):
    def __init__(self):
        self.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        super().__init__()

    def respond(self, user_prompt, system_prompt = None, argument = None):
        response = None
        while response is None:
            try:
                if argument:
                    completion = client_anthropic.messages.create(
                        model=self.model,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": f"{user_prompt}"}
                        ],
                        max_tokens=1024,
                    )
                    response = completion.content[0].text
                    break
                else:
                    completion = client_anthropic.messages.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": f"{user_prompt}"}
                        ],
                        max_tokens=1024,
                    )
                    response = completion.content[0].text
                    break
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        if argument:
            parsed_response = self.parse_response_argument(user_prompt, argument, response)
        else:
            parsed_response = self.parse_response(user_prompt, response)
        return response, parsed_response



class CommandRPlus(BaseModel, BaseModel_Scenario):
    def __init__(self):
        self.model = "command-r-plus"
        super().__init__()

    def respond(self, user_prompt, system_prompt = None, argument = None):
        response = None
        while response is None:
            try:
                if argument:
                    completion = client_cohere.chat(
                        model=self.model,
                        chat_history=[
                            {"role": "SYSTEM", "text": f"{system_prompt}"},
                        ],
                        message=f"{user_prompt}",
                        seed=1
                    )
                    response = completion.text
                    break
                else:
                    completion = client_cohere.chat(
                        model=self.model,
                        message=f"{user_prompt}",
                        seed=1
                    )
                    response = completion.text
                    break
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        if argument:
            parsed_response = self.parse_response_argument(user_prompt, argument, response)
        else:
            parsed_response = self.parse_response(user_prompt, response)
        return response, parsed_response



class Mistral8x7BInst(BaseModel):
    def __init__(self):
        self.model = "mistral.mixtral-8x7b-instruct-v0:1"
        super().__init__()

    def respond(self, user_prompt):
        body = {
                "prompt": f"<s>[INST] {user_prompt} [/INST]",
                "max_tokens": 1024,
                "temperature": 0,
            }
        
        response = None
        while response is None:
            try:
                results = client_bedrock.invoke_model(
                    modelId=self.model,
                    body=json.dumps(body)
                )
                response_body = json.loads(results["body"].read())
                outputs = response_body.get("outputs")
                response = outputs[0]['text']
                break
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        parsed_response = self.parse_response(user_prompt, response)
        return response, parsed_response



class LLaMA2_70BChat(BaseModel):
    def __init__(self):
        self.model = "meta.llama2-70b-chat-v1"
        super().__init__()

    def respond(self, user_prompt):
        body = {
                "prompt": f"<s>[INST] {user_prompt} [/INST]",
                "max_gen_len": 1024,
                "temperature": 0,
            }
        
        response = None
        while response is None:
            try:
                results = client_bedrock.invoke_model(
                    modelId=self.model,
                    body=json.dumps(body)
                )
                response_body = json.loads(results["body"].read())
                response = response_body["generation"]
                break
            except KeyboardInterrupt:
                sys.exit()
            except Exception as error:
                print(error)
        parsed_response = self.parse_response(user_prompt, response)
        return response, parsed_response