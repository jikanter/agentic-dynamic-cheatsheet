import ast
from datetime import datetime
import json
import os
import pandas as pd
import numpy as np

from datasets import load_dataset, load_from_disk
from tap import Tap
from dynamic_cheatsheet.language_model import LanguageModel, SUPPORTED_APPROACHES
from dynamic_cheatsheet.utils.evaluation import eval_for_GameOf24, eval_for_multiple_choice, eval_for_exact_matching_with_no_punctuation, eval_equation_balancer

from dotenv import load_dotenv

PREDEFINED_PROMPTS = {
    "GameOf24": "Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. Please present a single expression that evaluates to 24.",
}

# Approaches that require pre-computed embeddings for retrieval
RETRIEVAL_APPROACHES = ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "DynamicCheatsheet_CumulativeRetrieval"]


class Arguments(Tap):
    """
    Arguments for the Dynamic Cheatsheet benchmark runner.
    """
    # Task name
    task: str = "GameOf24"

    # Approach name
    approach_name: str = "DynamicCheatsheet_Cumulative"

    # Model name (format: "provider/model", e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250514")
    model_name: str = "openai/gpt-4o-mini"

    # Paths to the prompt files
    generator_prompt_path: str = "prompts/generator_prompt.txt"
    cheatsheet_prompt_path: str = None

    # Additional model-related arguments
    max_tokens: int = 2048
    temperature: float = 0.0
    max_num_rounds: int = 1

    execute_python_code: bool = True
    initialize_cheatsheet_path: str = None
    retrieve_top_k: int = 3
    reasoning_effort: str = None  # OpenAI reasoning effort: "low", "medium", or "high"
    use_code_interpreter: bool = False  # Use OpenAI Code Interpreter (Responses API) instead of local subprocess

    # Continue from the previous run
    continue_from_last_run_path: str = None

    # Additional save-path-related arguments
    save_directory: str = "results"
    additional_flag_for_save_path: str = ""
    max_n_samples: int = -1
    no_shuffle: bool = False


def read_file(file_path: str) -> str:
    """
    Read the file and return the content.
    """
    with open(file_path, "r") as file:
        return file.read()


def write_jsonl(file_path, data):
    """
    Save the outputs to a JSONL file.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")


def main(args: Arguments):
    """
    Main function to run the benchmark.
    """
    # Load the environment variables
    load_dotenv("config.env")

    # Validate approach name
    if args.approach_name not in SUPPORTED_APPROACHES:
        raise ValueError(
            f"Unknown approach '{args.approach_name}'. "
            f"Supported approaches: {', '.join(SUPPORTED_APPROACHES)}"
        )

    # Read the prompt files
    args.generator_prompt = read_file(args.generator_prompt_path)
    if args.cheatsheet_prompt_path:
        args.cheatsheet_prompt = read_file(args.cheatsheet_prompt_path)
    else:
        args.cheatsheet_prompt = "(empty)"

    args.max_n_samples = int(args.max_n_samples)

    # Build extra API parameters (e.g., reasoning effort for OpenAI reasoning models)
    extra_api_params = {}
    if args.reasoning_effort:
        extra_api_params["reasoning_effort"] = args.reasoning_effort

    # Initialize the language model
    model = LanguageModel(
        model_name=args.model_name,
        extra_api_params=extra_api_params if extra_api_params else None,
    )

    # Create a code interpreter session if requested (one per benchmark run)
    if args.use_code_interpreter:
        container_id = model.create_container()
        print(f"Code interpreter session ready (id={container_id})")

    # Add a flag to the save path if the code execution is not allowed
    if not args.execute_python_code:
        args.additional_flag_for_save_path += "_no-code-execution"

    # Load the dataset based on the task name
    if args.task in PREDEFINED_PROMPTS and args.task != "P3_Test":
        dataset = load_dataset("turingmachine/meta-prompting")
        dataset = dataset[args.task]
    elif args.task in ["GPQA_Diamond", "AIME_2020_2024", "AIME_2024", "AIME_2025", "MMLU_Pro_Physics", "MMLU_Pro_Engineering", "MathEquationBalancer"]:
        dataset = load_from_disk(f"data/{args.task}")
    else:
        raise ValueError(f"Task {args.task} is not recognized. Please make sure the task name is correct.")

    # If the previous run parameter is provided, make sure that the provided arguments are consistent with those found in the previous run
    if args.continue_from_last_run_path:
        if not os.path.exists(args.continue_from_last_run_path):
            raise ValueError(f"The provided path {args.continue_from_last_run_path} does not exist.")

        # Read the previous run parameters from the previous run file and compare them with the provided arguments
        previous_run_param_path = args.continue_from_last_run_path.replace(".jsonl", "_params.json")
        with open(previous_run_param_path, "r") as file:
            previous_run_params = json.load(file)

        # Compare the provided arguments with the previous run parameters
        # Note: cheatsheet_prompt_path was previously named cheatshet_prompt_path (typo); handle both
        args_keys = ["generator_prompt_path", "temperature", "execute_python_code", "task", "model_name", "approach_name", "max_num_rounds"]

        for key in args_keys:
            if getattr(args, key) != previous_run_params.get(key):
                raise ValueError(f"The provided argument '{key}' is inconsistent with the previous run. Previous value: {previous_run_params.get(key)}, current value: {getattr(args, key)}.")

        # Create a new save path name based on the previous run path
        args.save_path_name = args.continue_from_last_run_path.replace(".jsonl", "_continued.jsonl")
    else:
        # Create a new save path name based on the current time stamp
        time_stamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
        args.save_path_name = f"{args.save_directory}/{args.task}/{args.model_name}_{args.approach_name}_{time_stamp}_{args.additional_flag_for_save_path}.jsonl"

        # Create the directory if it does not exist
        dir_path = os.path.dirname(args.save_path_name)
        os.makedirs(dir_path, exist_ok=True)

    save_param_path = args.save_path_name.replace(".jsonl", "_params.json")
    dir_path = os.path.dirname(save_param_path)
    os.makedirs(dir_path, exist_ok=True)

    # Save the arguments to a file
    with open(save_param_path, "w") as file:
        json.dump(args.as_dict(), file, indent=4)

    # Initialize the cheatsheet
    cheatsheet = "(empty)"
    if args.initialize_cheatsheet_path is not None:
        with open(args.initialize_cheatsheet_path, "r") as file:
            cheatsheet = file.read()

    # Initialize the outputs and the generator outputs so far
    outputs = []
    generator_outputs_so_far = []
    if args.continue_from_last_run_path:
        # Load the previous run
        with open(args.continue_from_last_run_path, "r") as file:
            outputs = [json.loads(line) for line in file.readlines()]

        # Load the previous cheatsheet from the last output (if available)
        last_cheatsheet = outputs[-1].get("final_cheatsheet")
        if last_cheatsheet is not None:
            cheatsheet = last_cheatsheet

        generator_outputs_so_far = [output["final_output"] for output in outputs]

        # Print the details
        print(f"Continuing from the previous run at {args.continue_from_last_run_path}.")
        print(f"Loaded {len(outputs)} examples from the previous run.")
        print(f"Most recent cheatsheet: {cheatsheet[:200]}...")
        print("-" * 50)

    # Shuffle the dataset if the no_shuffle flag is not set
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=10)

    # Initialize the questions and the embeddings
    questions = None
    embeddings = None
    if args.approach_name in RETRIEVAL_APPROACHES:
        embeddings_path = f"embeddings/{args.task}.csv"
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found at '{embeddings_path}'. "
                f"Pre-computed embeddings are required for the '{args.approach_name}' approach."
            )
        df = pd.read_csv(embeddings_path)
        questions = df["input"].tolist()
        embeddings = df["embedding"].apply(ast.literal_eval)
        embeddings = np.array(embeddings.tolist())  # (N, embedding_dim)

        # Re-order the embeddings based on the order of the dataset inputs
        dataset_inputs = [example["input"] for example in dataset]
        indices = [questions.index(inp) for inp in dataset_inputs]
        embeddings = embeddings[indices]
        questions = dataset_inputs
    elif args.approach_name == "FullHistoryAppending":
        # FullHistoryAppending doesn't need embeddings, just the input corpus
        questions = [example["input"] for example in dataset]
    else:
        questions = [example["input"] for example in dataset]

    start_idx = len(outputs)
    correct_so_far = 0
    total_so_far = 0

    # Iterate over the dataset
    for idx, example in enumerate(dataset):
        original_input = dataset[idx]["input"]
        original_target = dataset[idx]["target"]
        orig_input = example["input"]
        if args.task in PREDEFINED_PROMPTS:
            current_input = f"{PREDEFINED_PROMPTS[args.task]}\n\nQuestion #{idx+1}:\n{orig_input}"
        else:
            current_input = f"Question #{idx+1}:\n{orig_input}"

        if args.task in ["AIME_2020_2024", "AIME_2024", "AIME_2025"]:
            current_input = f"{current_input} (Please provide your answer in the form of an integer, e.g., 1234, with no Markdown formatting or additional text; make sure to pay attention to the desired format of the final answer though.)"
        elif args.task == "MathEquationBalancer":
            current_input = f"Below is an equation with missing operators. Your task is to fill in the blanks with the correct mathematical operators: +, -, *, or /. Ensure that the equation is correct once the operators are added. The operators should be placed in the sequence they appear from left to right. Include the full equation with the operators filled in. For instance, for the equation 1 ? 2 ? 3 = 6, the correct answer is 1 + 2 + 3 = 6.\n\nEquation: {current_input}"

        # Skip the examples that have been already seen in the previous run
        if idx < start_idx:
            continue

        # Print the details
        print(f"### Example {idx+1} ###")

        # Generate the output from the language model
        output_dict = model.advanced_generate(
            approach_name=args.approach_name,
            input_txt=current_input,
            cheatsheet=cheatsheet,
            generator_template=args.generator_prompt,
            cheatsheet_template=args.cheatsheet_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_num_rounds=args.max_num_rounds,
            allow_code_execution=args.execute_python_code,
            code_execution_flag="EXECUTE CODE!",
            original_input_corpus=questions[:idx+1],
            original_input_embeddings=embeddings[:idx+1] if args.approach_name in RETRIEVAL_APPROACHES else None,
            generator_outputs_so_far=generator_outputs_so_far,
            retrieve_top_k=args.retrieve_top_k,
            use_code_interpreter=args.use_code_interpreter,
        )

        generator_outputs_so_far.append(output_dict["final_output"])

        outputs.append({
            "input": current_input,
            "target": original_target,
            "raw_input": original_input,
            **output_dict,
        })
        cheatsheet = output_dict["final_cheatsheet"]
        final_answer = output_dict["final_answer"]

        print(f"@ CHEATSHEET:\n{cheatsheet}")
        print('- ' * 50)
        print(f"Input: {current_input}")
        print(f"Target: {original_target}")
        print(f"Final answer: {final_answer}")
        print("**" * 50)

        if args.task == "GameOf24":
            result = eval_for_GameOf24(original_input, final_answer)
        elif args.task in ["AIME_2025", "AIME_2024", "AIME_2020_2024"]:
            result = eval_for_exact_matching_with_no_punctuation(final_answer.lower(), original_target.lower())
        elif args.task in ["GPQA_Diamond", "MMLU_Pro_Engineering", "MMLU_Pro_Physics"]:
            result = eval_for_multiple_choice(current_input, final_answer, original_target)
        elif args.task == "MathEquationBalancer":
            result = eval_equation_balancer(None, final_answer, original_target)
        else:
            raise ValueError(f"Task {args.task} not supported.")

        if result:
            correct_so_far += 1
        total_so_far += 1

        print(f"---- Correct so far: {correct_so_far}/{total_so_far}")
        print("###" * 50)

        # Save the outputs to a file after each example (for crash recovery)
        write_jsonl(args.save_path_name, outputs)

        if args.max_n_samples > 0 and idx == args.max_n_samples - 1:
            break

    # Save the final outputs
    write_jsonl(args.save_path_name, outputs)

    # Print final summary
    if total_so_far > 0:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS: {correct_so_far}/{total_so_far} correct ({100*correct_so_far/total_so_far:.1f}%)")
        print(f"Results saved to: {args.save_path_name}")
        print(f"{'='*60}")


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
