import os
import uuid
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future
from openai import OpenAI
import ollama
from constants import system_prompt, system_prompt_v2, system_prompt_v3

load_dotenv()


# Prompt template for fusing summary with information received from other agents
INTERACTION_PROMPT = """Integrate [NEW INFO] into [PRIOR SUMMARY] and return the updated summary only.

Rules:
- Every detail already in [PRIOR SUMMARY] must be preserved verbatim or made more specific. Never drop or vague-ify existing details.
- Only add content from [NEW INFO] that is not already covered.
- If [NEW INFO] adds nothing new, return [PRIOR SUMMARY] exactly as-is.
- If both are empty, return an empty string and nothing else.

[PRIOR SUMMARY]:
{summary}

[NEW INFO]:
{received_info}

"""

# Prompt template for fusing summary with privately discovered information from sites
PRIVATE_INFO_PROMPT = """Integrate [NEW INFO] into [PRIOR SUMMARY] and return the updated summary only.

Rules:
- Every detail already in [PRIOR SUMMARY] must be preserved verbatim or made more specific. Never drop or vague-ify existing details.
- Only add content from [NEW INFO] that is not already covered.
- If [NEW INFO] adds nothing new, return [PRIOR SUMMARY] exactly as-is.
- If both are empty, return an empty string and nothing else.

[PRIOR SUMMARY]:
{summary}

[NEW INFO]:
{private_info}

"""


class LLMRequestError(RuntimeError):
    """Raised when the LLM provider returns an unrecoverable response."""
    pass


# Global executor shared across all LLM instances
_LLM_EXECUTOR: ThreadPoolExecutor | None = None


class LLM:
    def __init__(self, agent=None):
        self.agent = agent
        
        # Provider selection: "openai" | "vllm" | "ollama"
        self.provider = os.getenv("LLM_PROVIDER", "vllm").lower()
        
        # Default models per provider
        default_models = {
            "openai": "gpt-4o",
            "vllm":   "google/gemma-3-4b-it",
            "ollama": "gemma4:e4b",
        }
        self.model = os.getenv("LLM_MODEL", default_models.get(self.provider, "gpt-4o"))
        
        if self.provider in ("openai", "vllm"):
            # vLLM is OpenAI-compatible; switch backends by changing base_url/api_key.
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY" if self.provider == "openai" else "VLLM_API_KEY", "EMPTY"),
                base_url=os.getenv("OPENAI_BASE_URL" if self.provider == "openai" else "VLLM_BASE_URL",
                                   "https://api.openai.com/v1" if self.provider == "openai" else "http://localhost:8000/v1"),
            )

        # Initialize global executor if needed
        global _LLM_EXECUTOR
        if _LLM_EXECUTOR is None:
            max_workers = int(os.getenv("LLM_MAX_WORKERS", "200"))
            _LLM_EXECUTOR = ThreadPoolExecutor(
                max_workers=max_workers, 
                thread_name_prefix="llm"
            )

        self._executor = _LLM_EXECUTOR
        self._futures: dict[str, Future] = {}

        # Persistent conversation history per agent
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Context growth tracking: one entry per successful LLM call.
        # prompt_tokens reflects the full context sent (all prior turns + new message).
        self.context_growth: list[dict] = []

        print(f"🤖 LLM initialized: provider={self.provider}, model={self.model}")

    def chat(self, prompt: str) -> str:
        """Make a chat completion request."""
        if self.provider == "ollama":
            return self._ollama_chat(prompt)
        return self._openai_compatible_chat(prompt)

    def _openai_compatible_chat(self, prompt: str) -> str:
        """chat.completions request — works for both OpenAI and vLLM."""
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                timeout=200,
            )
            reply = response.choices[0].message.content or ""
            if response.usage:
                self.context_growth.append({
                    "call": len(self.context_growth) + 1,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                })
            # Reset to stateless: each fusion call is independent, so prior
            # conversation history only causes summary degradation over time.
            self.messages = [self.messages[0]]
            return reply
        except Exception as exc:
            self.messages.pop()
            raise LLMRequestError(f"LLM chat request failed: {exc}") from exc

    def _ollama_chat(self, prompt: str) -> str:
        """Make a chat request using native Ollama library."""
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
            )
            reply = response.message.content or ""
            prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
            completion_tokens = getattr(response, "eval_count", 0) or 0
            self.context_growth.append({
                "call": len(self.context_growth) + 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            # Reset to stateless: each fusion call is independent, so prior
            # conversation history only causes summary degradation over time.
            self.messages = [self.messages[0]]
            return reply
        except Exception as exc:
            self.messages.pop()
            raise LLMRequestError(f"Ollama chat request failed: {exc}") from exc

    def submit_interaction(self, summary: str, received_info: str) -> str:
        """
        Submit a fusion task for summary + received information from other agents.
        Returns a task_id to poll for results.
        """
        # Skip LLM call if there is nothing real to merge
        if not summary and not received_info:
            task_id = uuid.uuid4().hex
            future = self._executor.submit(lambda: "")
            self._futures[task_id] = future
            return task_id
        prompt = INTERACTION_PROMPT.format(
            summary=summary or "(no prior knowledge)",
            received_info=received_info or "(no received information)"
        )
        task_id = uuid.uuid4().hex
        future = self._executor.submit(self.chat, prompt)
        self._futures[task_id] = future
        return task_id

    def submit_private_info(self, summary: str, private_info: str) -> str:
        """
        Submit a fusion task for summary + privately discovered information from sites.
        Returns a task_id to poll for results.
        """
        # Skip LLM call if there is nothing real to merge
        if not summary and not private_info:
            task_id = uuid.uuid4().hex
            future = self._executor.submit(lambda: "")
            self._futures[task_id] = future
            return task_id
        prompt = PRIVATE_INFO_PROMPT.format(
            summary=summary or "(no prior knowledge)",
            private_info=private_info or "(no new information)"
        )
        task_id = uuid.uuid4().hex
        future = self._executor.submit(self.chat, prompt)
        self._futures[task_id] = future
        return task_id

    def poll(self) -> list[tuple[str, str]]:
        """Check for completed tasks and return their results."""
        completed: list[tuple[str, str]] = []
        to_remove: list[str] = []
        fatal_exc: Exception | None = None

        for task_id, fut in list(self._futures.items()):
            if fut.done():
                to_remove.append(task_id)
                try:
                    result = fut.result()
                    completed.append((task_id, result))
                except LLMRequestError as exc:
                    fatal_exc = exc
                except Exception as exc:
                    completed.append((task_id, f"LLM error: {exc}"))

        for task_id in to_remove:
            self._futures.pop(task_id, None)

        if fatal_exc is not None:
            raise fatal_exc

        return completed

    def get_context_stats(self) -> dict:
        """Return context growth trajectory and final context size for this agent."""
        final_context_size = self.context_growth[-1]["prompt_tokens"] if self.context_growth else 0
        return {
            "final_context_size": final_context_size,
            "num_calls": len(self.context_growth),
            "context_growth": self.context_growth,
        }

    def cancel_all(self) -> None:
        """Cancel all pending tasks."""
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()
