# llm_client.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

class GraniteClient:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        if self.device == 'cpu':
            print("[LLM] ⚠️ CUDA not available. Running on CPU.")
        else:
            print(f"[LLM] ✅ Running on GPU: {torch.cuda.get_device_name(0)}")

        self.model_name = "ibm-granite/granite-3.2-2b-instruct"
        print(f"[LLM] Loading model -> {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if (self.device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None,
            low_cpu_mem_usage=True,
        )

        if self.device == 'cpu':
            self.model = self.model.to("cpu")

        print("[LLM] ✅ Granite loaded successfully")

    def _clean_answer(self, full_resp: str) -> str:
        # Extract after "Answer:" (case-insensitive)
        parts = re.split(r"(?i)\banswer\s*:", full_resp, maxsplit=1)
        answer = parts[1].strip() if len(parts) > 1 else full_resp.strip()

        # Trim if model restarts prompt blocks
        for token in ["Context:", "Question:", "Answer:"]:
            if token in answer:
                answer = answer.split(token)[0].strip()

        # Guard: if it echoed the instruction text, force not-found
        if answer.lower().startswith("you are a strict academic"):
            return "Information not found in the document."

        # Guard: too short → not useful
        if len(answer.split()) < 2:
            return "Information not found in the document."

        return answer

    def generate_answer(self, question: str, context: str, max_new_tokens: int = 100) -> str:
        print("\n" + "="*90)
        print("[LLM] 🧠 Generating Answer...")
        print(f"[LLM] ❓ Question: {question}")

        prompt = f"""
You are a strict academic QA model. Answer ONLY from the context.
If the answer is missing, reply ONLY with:
"Information not found in the document."

Respond in 3–5 concise sentences.
Do not add external knowledge.

Context:
{context}

Question: {question}

Answer:
""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )

        if self.device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self._clean_answer(full_resp)

        print("[LLM] ✅ Answer Generated")
        print("="*90 + "\n")
        return answer
