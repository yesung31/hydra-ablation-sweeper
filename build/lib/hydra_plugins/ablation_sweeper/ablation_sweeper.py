import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Override
from hydra.core.utils import JobReturn
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction

log = logging.getLogger(__name__)

@dataclass
class AblationSweeperConf:
    _target_: str = "hydra_plugins.ablation_sweeper.ablation_sweeper.AblationSweeper"
    params: Optional[Dict[str, Any]] = None
    cartesian_params: Optional[List[str]] = None
    max_batch_size: Optional[int] = None

ConfigStore.instance().store(
    group="hydra/sweeper",
    name="ablation",
    node=AblationSweeperConf,
    provider="ablation_sweeper",
)

class AblationSweeper(Sweeper):
    def __init__(
        self,
        params: Optional[Dict[str, Any]],
        cartesian_params: Optional[List[str]] = None,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.params = params or {}
        self.cartesian_params = cartesian_params or []
        self.max_batch_size = max_batch_size
        
        self.hydra_context: Optional[HydraContext] = None
        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.overrides: Optional[List[List[List[str]]]] = None
        self.batch_index = 0

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        from hydra.core.plugins import Plugins
        self.hydra_context = hydra_context
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    def sweep(self, arguments: List[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        print(f"DEBUG: Entering sweep with arguments: {arguments}")
        # Merge params from config and arguments
        params_conf = []
        for k, v in self.params.items():
            params_conf.append(f"{k}={v}")
        params_conf.extend(arguments)

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)

        # Generate all combinations
        self.overrides = self._generate_overrides(overrides)
        print(f"DEBUG: Generated {len(self.overrides)} batches")
        
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        returns: List[Sequence[JobReturn]] = []
        initial_job_idx = 0
        
        while self.batch_index < len(self.overrides):
            batch = self.overrides[self.batch_index]
            self.batch_index += 1
            
            self.validate_batch_is_legal(batch)
            results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)
            
            for r in results:
                _ = r.return_value
            
            initial_job_idx += len(batch)
            returns.append(results)
            
        return returns

    def _generate_overrides(self, overrides: List[Override]) -> List[List[List[str]]]:
        cartesian_elements: Dict[str, List[str]] = {}
        ablation_sweep_elements: Dict[str, List[str]] = {}
        fixed_elements: List[str] = []
        
        # We need to know which keys are in the sweep to properly deduplicate against base config
        sweep_keys = set()

        for override in overrides:
            key = override.get_key_element()
            if key in self.cartesian_params:
                if override.is_sweep_override():
                    cartesian_elements[key] = [f"{key}={val}" for val in override.sweep_string_iterator()]
                else:
                    cartesian_elements[key] = [f"{key}={override.get_value_element_as_str()}"]
                sweep_keys.add(key)
            elif override.is_sweep_override():
                ablation_sweep_elements[key] = [f"{key}={val}" for val in override.sweep_string_iterator()]
                sweep_keys.add(key)
            else:
                fixed_elements.append(f"{key}={override.get_value_element_as_str()}")
                # Fixed elements are also effectively swept to a single value
                sweep_keys.add(key)

        # To deduplicate against base config, we need to know the values in the base config
        # for all keys we are touching.
        base_config_values = {}
        if self.config:
            for key in sweep_keys:
                try:
                    val = OmegaConf.select(self.config, key, throw_on_missing=True)
                    if val is not None:
                        # Convert to string to match override format
                        base_config_values[key] = str(val)
                except Exception:
                    # Key might not be in config, which is fine
                    pass

        # Cartesian product of cartesian_params
        if cartesian_elements:
            keys = sorted(cartesian_elements.keys())
            cartesian_products = [list(x) for x in itertools.product(*[cartesian_elements[k] for k in keys])]
        else:
            cartesian_products = [[]]

        all_jobs: List[List[str]] = []

        # 1. Base case: Cartesian products of cartesian_params + fixed_elements
        for cp in cartesian_products:
            all_jobs.append(cp + fixed_elements)

        # 2. Ablation cases: For each ablation parameter that is a sweep, vary it
        for key, values in ablation_sweep_elements.items():
            for val in values:
                for cp in cartesian_products:
                    all_jobs.append(cp + fixed_elements + [val])

        # Remove potential duplicates by evaluating what the final config would look like for these keys
        unique_jobs = []
        seen_configs = set()
        
        for job in all_jobs:
            # Merge job overrides and base config to find final values for this job
            current_job_values = base_config_values.copy()
            for override_str in job:
                k, v = override_str.split("=", 1)
                current_job_values[k] = v
            
            # Create a stable representation for deduplication
            # Include all keys that are present in either any job or base config for those keys
            config_identity = []
            for k in sorted(current_job_values.keys()):
                config_identity.append((k, current_job_values[k]))
            
            config_identity_tuple = tuple(config_identity)
            if config_identity_tuple not in seen_configs:
                seen_configs.add(config_identity_tuple)
                unique_jobs.append(job)

        if self.max_batch_size is None or self.max_batch_size == -1:
            return [unique_jobs]
        else:
            return [unique_jobs[i : i + self.max_batch_size] for i in range(0, len(unique_jobs), self.max_batch_size)]
