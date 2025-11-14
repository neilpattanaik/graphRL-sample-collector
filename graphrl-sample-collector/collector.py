from collections import defaultdict
import os
import pprint
from typing import Any, TypedDict
import argparse

from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from ray.experimental.tqdm_ray import tqdm

from nemo_rl.models.policy import PolicyConfig
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.response_datasets import DAPOMath17KDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataProcessFnCallable
from nemo_rl.data.llm_message_utils import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import Logger, LoggerConfig


TokenizerType = PreTrainedTokenizerBase

class RunConfig(TypedDict):
    prompt_file: str
    output_dir: str

    seed: int

    num_prompts_per_step: int
    num_rollouts_per_prompt: int
    num_batches: int


class RolloutCollectorConfig(TypedDict):
    run_config: RunConfig
    env: dict[str, Any]
    cluster: ClusterConfig
    policy: PolicyConfig


class RolloutDatasetCollector:
    def __init__(
        self,
        model_name: str,
        policy_generation: VllmGeneration,
        max_seq_len: int,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        num_rollouts_per_prompt: int,
        logger: Logger,
    ):
        self.model_name = model_name
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.max_seq_len = max_seq_len
        self.num_rollouts_per_prompt = num_rollouts_per_prompt
        self.logger = logger    

    def collect_rollout_results_for_batch(
        self, batch: BatchedDataDict[DatumSpec]
    ) -> BatchedDataDict:
        
        if self.num_rollouts_per_prompt > 1:
            batch = batch.repeat_interleave(self.num_rollouts_per_prompt)

        batch, _ = run_multi_turn_rollout(
            policy_generation=self.policy_generation,
            input_batch=batch,
            tokenizer=self.tokenizer,
            task_to_env=self.task_to_env,
            max_seq_len=self.max_seq_len,
            max_rollout_turns=1,
            greedy=False,
        )

        res = BatchedDataDict({
            "prompt": [log[0]["content"] for log in batch["message_log"]],
            "completion": [log[1]["content"] for log in batch["message_log"]],
            "reward": batch["total_reward"].tolist(),
            "gen_len": [log[1]["token_ids"].shape[0] for log in batch["message_log"]],
        })
        
        return res


    def run_collect_rollouts(self, dataloader: DataLoader, num_batches: int = -1) -> None:
        """Collect rollouts and save them to output files."""
        num_batches = len(dataloader) if num_batches == -1 else num_batches
        self.policy_generation.prepare_for_generation()
        
        pbar = tqdm(
            dataloader,
            desc="Processing batches...",
            total=num_batches,
        )
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= num_batches:
                break

            results = self.collect_rollout_results_for_batch(batch)
            self.logger.log_batched_dict_as_jsonl(results, f"data_batch_idx{batch_idx}.jsonl")


        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
    )
    args, overrides = parser.parse_known_args()
    return args, overrides





def main():
    """Main entry point for rollout data collection."""
    # ---------------------------------------------------------------------------- #
    #                                  Arg Parsing                                 #
    # ---------------------------------------------------------------------------- #
    args, overrides = parse_args()
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    print("\nFinal config:")
    pprint.pprint(config)

    env_config = config["env"]["math"]
    cluster_config = config["cluster"]
    policy_config = config["policy"]
    run_config = config["run_config"]
    set_seed(run_config["seed"])

    _log_cfg = LoggerConfig(
        {
            "log_dir": run_config["output_dir"],
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False
        }
    )
    logger = Logger(_log_cfg)   # we only use this for json dumps

    # ---------------------------------------------------------------------------- #
    #                                 Data Loading                                 #
    # ---------------------------------------------------------------------------- #
    print("\n▶ Setting up tokenizer...")
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    print("  ✓ Tokenizer loaded")


    print("\n▶ Setting up dataset...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=run_config["prompt_file"],
    )
    data = DAPOMath17KDataset(seed=run_config["seed"])
    print("  ✓ Loaded DAPOMath17k dataset")

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (math_task_spec, math_hf_data_processor))
    )
    task_data_processors["math"] = (math_task_spec, math_hf_data_processor)

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=1024,
    )
    print(f"  ✓ Processed dataset with {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=run_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=False,
        num_workers=1,
    )
    print(f"  ✓ Dataloader created with batch size {run_config['num_prompts_per_step']}")
    
    # ---------------------------------------------------------------------------- #
    #                                   Ray Setup                                  #
    # ---------------------------------------------------------------------------- #
    print("\n▶ Initializing Ray...")
    init_ray()
    print("  ✓ Ray initialized")

    print("\n▶ Setting up math environment...")
    math_env = MathEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_config)
    task_to_env = {"math":math_env}
    print("  ✓ Math environment initialized")

    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="rollout-collector-cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(
        f"  ✓ Cluster initialized with {cluster_config['num_nodes']} nodes, "
        f"{cluster_config['gpus_per_node']} GPUs per node"
    )
    # ---------------------------------------------------------------------------- #
    #                               Policy Generation                              #
    # ---------------------------------------------------------------------------- #
    print("\n▶ Setting up policy generation...")
    generation_config = configure_generation_config(
        policy_config["generation"], tokenizer, is_eval=True
    )

    policy_generation = VllmGeneration(
        cluster=cluster,
        config=generation_config,
        name_prefix="vllm",
    )
    policy_generation.finish_generation()
    print("  ✓ Generation interface initialized")

    # ---------------------------------------------------------------------------- #
    #                            Collector Setup and Run                           #
    # ---------------------------------------------------------------------------- #
    collector = RolloutDatasetCollector(
        model_name=policy_config["model_name"],
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=policy_config["max_total_sequence_length"],
        num_rollouts_per_prompt=run_config["num_rollouts_per_prompt"],
        logger=logger,
    )

    print("\n▶ Starting rollout data collection...")
    collector.run_collect_rollouts(
        dataloader,
        num_batches=run_config["num_batches"],
    )
    print("  ✓ Rollout data collection completed! Exiting.")


if __name__ == "__main__":
    main()
