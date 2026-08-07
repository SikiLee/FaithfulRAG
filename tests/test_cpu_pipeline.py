import asyncio
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

from faithfulrag import FaithfulRAG
from faithfulrag.llm.backend import LLMBackend


class FakeSentenceTransformer:
    def __init__(self, _model_name):
        pass

    def encode(self, texts, convert_to_tensor=True, show_progress_bar=False):
        values = [[float(len(text.split()) + 1), float(sum(map(ord, text)) % 97 + 1)] for text in texts]
        return torch.tensor(values, dtype=torch.float32)


async def mock_complete(prompt, system_prompt=None, history_messages=None, model_name=None, **kwargs):
    if "identify the factual knowledge required" in prompt:
        return "1. Faster rotation affects a planet.\n2. Gravity is the requested effect."
    if "generate a background document" in prompt.lower():
        return "Faster rotation changes a planet. The supplied claim links rotation with stronger gravity."
    if "Extract factual statements" in prompt:
        return "1. Faster rotation changes a planet.\n2. Rotation is linked with stronger gravity."
    return '{"Reason": "The aligned evidence states the effect.", "Answer": "stronger gravity"}'


class CpuPipelineTest(unittest.TestCase):
    def test_end_to_end_control_flow_with_mock_llm(self):
        dataset = Dataset.from_list(
            [
                {
                    "id": "sample-1",
                    "question": "What happens to gravity?",
                    "answer": "stronger gravity",
                    "choices": ["weaker gravity", "stronger gravity"],
                    "context": "The experiment states that gravity becomes stronger. Rotation changed after impact.",
                }
            ]
        )
        LLMBackend.BACKENDS["mock"] = mock_complete
        with patch("faithfulrag.modules.SentenceTransformer", FakeSentenceTransformer), patch(
            "faithfulrag.modules.nltk.sent_tokenize",
            side_effect=lambda text: [part.strip() + "." for part in text.split(".") if part.strip()],
        ):
            rag = FaithfulRAG(
                backend_type="mock",
                model_name="mock-model",
                similarity_model="mock-embedding",
            )

            async def run():
                facts = await rag.get_self_facts(dataset)
                chunks = rag.get_topk_chunks(dataset, facts, sent_topk=2, chunk_topk=2, chunk_size=20)
                predictions = await rag.get_predictions(dataset, chunks, generation_type="scheduled_cot")
                return facts, chunks, predictions

            facts, chunks, predictions = asyncio.run(run())
            metrics = rag.evaluate(dataset, predictions, cot_format=True, detailed_output=True)
        self.assertTrue(facts[0]["facts"])
        self.assertTrue(chunks[0]["topk_chunks"])
        self.assertEqual(metrics["exact_match"], 100.0)
        self.assertEqual(metrics["acc"], 100.0)


if __name__ == "__main__":
    unittest.main()
