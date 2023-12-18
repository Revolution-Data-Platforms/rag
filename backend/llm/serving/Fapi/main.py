import torch, os, transformers, uvicorn, argparse, json
from fastapi import FastAPI
from dotenv import load_dotenv 
from pydantic import BaseModel    

# Load environment variables
parser = argparse.ArgumentParser()
parser.add_argument('--env', default='.env')
args = parser.parse_args()

load_dotenv(dotenv_path=args.env)

# Load environment variables
architecture = os.environ.get('architecture')
log_path = os.environ.get('log_path')
load_in_8bit = os.environ.get('load_model_in_8bits')

print(architecture)

# Setting up options
db_path = f'{log_path}/{architecture}.txt'


from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
)






# should be logging
print(f'Loading model...{architecture}')

# should be logging
def write_to_file(msg):
    f = open(db_path, "a")
    f.write(msg)
    f.close()

# from attention_sinks import AutoModelForCausalLM
# Loading model and tokenizer
model = transformers.AutoModelForCausalLM.from_pretrained(
    architecture,
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    architecture,
    trust_remote_code=False
)

# Default generation parameters
db_path = f'{log_path}/{architecture}.txt'

generation_config = model.generation_config
generation_config.max_new_tokens = 128
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config_default_dict = generation_config.to_dict()

# Set up FastAPI
app = FastAPI()

@app.get('/info')
def model_info():
    """Return model information, version, how to call"""
    return {
        "name": architecture,
        "version": 1.0
    }

@app.get('/health')
def service_health():
    """Return service health"""
    return {
        "status": "healthy"
    }

@app.post("/answer", )
def generate(question: str, gen_config):
    gen_config = json.loads(gen_config)
    if gen_config == {}:
        conf = generation_config_default_dict
    else:
        conf = {**generation_config_default_dict, **gen_config}

    inputs = tokenizer(question, return_tensors="pt", return_token_type_ids=False).to(model.device)
    
    with torch.inference_mode():
            generations = model.generate(
                 **inputs, 
                 generation_config=generation_config.from_dict(conf)
                 )
    return tokenizer.decode(generations[0], skip_special_tokens=True)



def serve():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    serve()
