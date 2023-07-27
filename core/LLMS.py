# %%
from typing import Any, List, Mapping, Optional
import requests, json
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class Remote_LLM(LLM):
    endpoint: str
    generation_config: dict

    @property
    def _llm_type(self) -> str:
        return "RDP Custom Remote Wrapper to LLM" 

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        
        headers = {
        'Content-Type': 'application/json',
        }
        data = {'question': prompt, 'gen_config': json.dumps(self.generation_config)}
        
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return requests.post(self.endpoint, params=data, headers=headers).json()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint": self.endpoint}

# %%
if __name__ == "__main__":
    
    endpoint = "http://mavrik.specgood.ai:8000/answer"
    question = "What is the meaning of life?"
    llm = Remote_LLM(endpoint=endpoint)
    llm(prompt='tell me a story')
    
# %%

# %%
