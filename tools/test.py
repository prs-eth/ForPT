from forpt.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
# NOTE: forpt.models must be imported BEFORE forpt.engines.test.
# engines/test.py imports forpt.models, which reaches engines/hooks/misc.py, which does
# `from forpt.engines.test import TESTERS`. When engines.test is the entry point -- i.e.
# when this script is run -- that re-entry lands in a still-partially-initialised module
# whose TESTERS (defined at engines/test.py:33, below its own model imports) does not exist
# yet, and the import dies with:
#     ImportError: cannot import name 'TESTERS' from partially initialized module
#                  'forpt.engines.test' (most likely due to a circular import)
# tools/train.py never hits this because it pulls models/hooks in first.
# Importing models here is safe: models/__init__.py binds build_model on its first line
# (before .default), so the re-entrant `from forpt.models import build_model` inside
# engines/test.py resolves against the partially-initialised package.
import forpt.models  # noqa: F401  -- import for side effect; order matters, see above
from forpt.engines.test import TESTERS
from forpt.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
