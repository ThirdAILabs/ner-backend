import re
import os
import tqdm
import random
from llm import OpenAILLM
from jinja2 import Environment
from typing import Optional, List, Dict, TypedDict
from prompts import extend_description_prompt as extend_description_template
from prompts import extend_examples_prompt as extend_examples_template
from prompts import context_prompt as context_prompt_template
from prompts import (
    annoated_data_generation_prompt as annotated_data_generation_prompt_template,
)

from utils import write_to_csv, requirements

env = Environment()
env.filters["random_sample"] = lambda lst, n: random.sample(lst, min(n, len(lst)))


class TagInfo(TypedDict):
    name: str
    desc: str
    examples: List[str]
    contexts: Optional[List[str]] = None


class DataFactory:
    def __init__(self, out_dir: str, openai_key: str, base_url: Optional[str] = None):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.llm = OpenAILLM(
            api_key=openai_key,
            base_url=base_url,
            track_usage_at=os.path.join(self.out_dir, "usage.json"),
        )

    def extend_description(self, tag: TagInfo) -> str:
        template = env.from_string(extend_description_template.user_prompt)
        response = self.llm.completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": extend_description_template.system_prompt,
                },
                {
                    "role": "user",
                    "content": template.render(
                        tag=tag,
                    ),
                },
            ],
            response_format=extend_description_template.response_format_json,
        )
        output = extend_description_template.ExtendedTagDescription.model_validate_json(
            response.choices[0].message.content
        )
        return output.extended_description

    def extend_examples(self, tag: TagInfo, k: int = 10) -> List[str]:
        template = env.from_string(extend_examples_template.user_prompt)
        response = self.llm.completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": extend_examples_template.system_prompt},
                {"role": "user", "content": template.render(tag=tag, k=k)},
            ],
            response_format=extend_examples_template.response_format_json,
        )
        output = extend_examples_template.ExtendedTagExamples.model_validate_json(
            response.choices[0].message.content
        )
        return output.extended_examples

    def get_tag_context(self, tag: TagInfo) -> List[str]:
        template = env.from_string(context_prompt_template.user_prompt)
        response = self.llm.completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": context_prompt_template.system_prompt},
                {
                    "role": "user",
                    "content": template.render(tag=tag, k=random.randint(20, 30)),
                },
            ],
            response_format=context_prompt_template.response_format_json,
        )

        output = context_prompt_template.TagContextScenarios.model_validate_json(
            response.choices[0].message.content
        )
        return output.scenarios

    def run_and_collect(
        self,
        messages: List[List[Dict[str, str]]],
        model: str = "gpt-4o",
        response_format: Optional[dict] = None,
        parallelize: bool = True,
    ) -> str:
        responses = []
        if parallelize:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor() as executor, tqdm.tqdm(
                total=len(messages), desc="Generating annotated data: "
            ) as pbar:
                futures = []

                # Submit arguments to the executor
                for msg in messages:
                    future = executor.submit(
                        self.llm.completion,
                        model=model,
                        messages=msg,
                        response_format=response_format,
                    )
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        response = future.result()
                        responses.append(response.choices[0].message.content)
                    except Exception as e:
                        print(f"Error processing message: {e}")
        else:
            for message in tqdm.tqdm(messages, desc="Generating annotated data: "):
                try:
                    response = self.llm.completion(
                        model=model, messages=message, response_format=response_format
                    )
                    responses.append(response.choices[0].message.content)
                except Exception as e:
                    pass

        return responses

    def generate(
        self,
        tags_info: List[TagInfo],
        k: int = 1000,
        user_instructions: List[str] = [],
        write_batch_size: int = 200,
        generate_per_llm_call: int = 25,
    ):
        write_batch_size = min(write_batch_size, k)
        generate_per_llm_call = min(generate_per_llm_call, k)

        output_file = os.path.join(self.out_dir, "generated_data.csv")

        # -----------(1) Extend the tag description, examples and get the tag contexts---------------
        for tag in tqdm.tqdm(tags_info, desc="Enhancing tag information: "):
            extended_desc = self.extend_description(tag)
            tag["desc"] = extended_desc
            tag["examples"].extend(self.extend_examples(tag, k=20))
            if tag.get("contexts") is None:
                tag["contexts"] = self.get_tag_context(tag)

        # ---------- (2) Generate Annotated Data ----------
        num_llm_calls_per_batch = write_batch_size // generate_per_llm_call
        for i in range(0, k, write_batch_size):
            messages = []
            user_content_template = env.from_string(
                annotated_data_generation_prompt_template.user_prompt
            )
            messages = [
                [
                    {
                        "role": "system",
                        "content": annotated_data_generation_prompt_template.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content_template.render(
                            k=min(generate_per_llm_call, k - i),
                            tagInfo=tags_info,
                            requirements=requirements,
                            user_instructions=user_instructions,
                        ),
                    },
                ]
                for _ in range(num_llm_calls_per_batch)
            ]
            responses = self.run_and_collect(
                messages,
                model="gpt-4o",
                response_format=annotated_data_generation_prompt_template.response_format_json,
                parallelize=True,
            )

            annotated_text = []
            for resp in responses:
                try:
                    annotated_text.extend(
                        annotated_data_generation_prompt_template.AnnotatedData.model_validate_json(
                            resp
                        ).sentences
                    )
                except Exception as e:
                    pass

            data_points = self.transform(
                annotated_text,
                [tag["name"].upper() for tag in tags_info],
                source_col="source",
                target_col="target",
            )

            if data_points:
                write_to_csv(
                    path=output_file,
                    data_points=data_points,
                    fieldnames=["source", "target"],
                )

    def transform_sentence(
        self, annotated_text: str, tags_name: List[str], source_col, target_col: str
    ) -> Dict[str, str]:
        annotated_text = annotated_text.strip()
        try:
            pattern = re.compile(
                r"(?P<before>[^\w\s#]*)"
                r"#+(?P<entity>[^#]+?)#+(?P<tag>[A-Z_]+)#+"
                r"(?P<after>[^\w\s#']*[\w']*)?"
            )
            source_tokens, target_tokens = [], []

            last_idx = 0
            for match in pattern.finditer(annotated_text):
                start, end = match.span()

                # Handle text before this match
                if last_idx < start:
                    prefix_text = annotated_text[last_idx:start]
                    prefix_tokens = prefix_text.strip().split()
                    source_tokens.extend(prefix_tokens)
                    target_tokens.extend(["O"] * len(prefix_tokens))

                # Process the matched token + surrounding punctuation
                before = match.group("before") or ""
                entity = match.group("entity").strip()
                tag = match.group("tag").upper()
                after = match.group("after") or ""

                full_span = (before + entity + after).strip()
                token_list = full_span.split()
                source_tokens.extend(token_list)

                if tag in tags_name:
                    target_tokens.extend([tag] * len(token_list))
                else:
                    target_tokens.extend(["O"] * len(token_list))

                last_idx = end

            # Handle remaining text
            if last_idx < len(annotated_text):
                suffix = annotated_text[last_idx:]
                suffix_tokens = suffix.strip().split()
                source_tokens.extend(suffix_tokens)
                target_tokens.extend(["O"] * len(suffix_tokens))

            return {
                source_col: " ".join(source_tokens),
                target_col: " ".join(target_tokens),
            }

        except Exception as e:
            pass
        return {
            source_col: " ".join(source_tokens),
            target_col: " ".join(target_tokens),
        }

    def transform(
        self,
        annotated_texts: List[str],
        tags_name: List[str],
        source_col: str = "source",
        target_col: str = "target",
    ) -> List[Dict[str, str]]:

        transformed_data = list(
            map(
                lambda x: self.transform_sentence(x, tags_name, source_col, target_col),
                annotated_texts,
            )
        )
        return list(
            filter(
                lambda x: x[source_col] and x[target_col],
                transformed_data,
            )
        )
