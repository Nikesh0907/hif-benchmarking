#!/usr/bin/env python3
"""Inventory of pretrained checkpoints already present in this repo.

This script is intentionally lightweight: it does not attempt to run any method.
It only provides a single source-of-truth list of where the weights live.

Usage:
  python tools/pretrained_inventory.py --list
  python tools/pretrained_inventory.py --validate
  python tools/pretrained_inventory.py --json

Exit codes:
  0: OK
  2: one or more referenced files are missing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import glob
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PretrainedArtifact:
	method: str
	family: str
	framework: str
	dataset_hint: str
	artifact_path: str
	notes: str = ""

	def abs_path(self) -> Path:
		return REPO_ROOT / self.artifact_path

	def _tf_data_shards_exist(self) -> bool:
		"""Best-effort check for TF v1 checkpoint completeness.

		If artifact_path points to a `.index` file, a matching `.data-*` shard
		must exist next to it for the checkpoint to be restorable.
		"""
		p = self.abs_path()
		if p.suffix != ".index":
			return True
		prefix = str(p.with_suffix(""))
		# Typical naming: <prefix>.data-00000-of-00001 (may have multiple shards)
		return len(glob.glob(prefix + ".data-*") ) > 0

	def exists(self) -> bool:
		p = self.abs_path()
		if not p.exists():
			return False
		# For TF checkpoints, require data shard(s) when checking an .index entry.
		if self.framework.lower() == "tensorflow":
			return self._tf_data_shards_exist()
		return True


ARTIFACTS: List[PretrainedArtifact] = [
	# HSISR (PyTorch)
	PretrainedArtifact(
		method="HSISR",
		family="DeepShare",
		framework="pytorch",
		dataset_hint="CAVE",
		artifact_path=(
			"methods/_HSISR/models/"
			"Cave_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Wed_Mar_31_03:00:46_2021.pth"
		),
		notes="31-band HSI SR/fusion model; see methods/_HSISR/mains.py",
	),
	PretrainedArtifact(
		method="HSISR",
		family="DeepShare",
		framework="pytorch",
		dataset_hint="Harvard",
		artifact_path=(
			"methods/_HSISR/models/"
			"Harvard_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Fri_Apr__2_15:35:55_2021.pth"
		),
		notes="31-band HSI SR/fusion model; see methods/_HSISR/mains.py",
	),

	# HSRnet (TensorFlow checkpoints)
	PretrainedArtifact(
		method="HSRnet",
		family="HSRnet",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_HSRnet/models(cave)/model-150000.ckpt.data-00000-of-00001",
		notes="TensorFlow v1-style checkpoint (needs matching .index + checkpoint file)",
	),
	PretrainedArtifact(
		method="HSRnet",
		family="HSRnet",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_HSRnet/models(cave)/model-150000.ckpt.index",
	),
	PretrainedArtifact(
		method="HSRnet",
		family="HSRnet",
		framework="tensorflow",
		dataset_hint="Harvard",
		artifact_path="methods/_HSRnet/models(harvard)/model-150000.ckpt.data-00000-of-00001",
		notes="TensorFlow v1-style checkpoint (needs matching .index/.meta + checkpoint file)",
	),
	PretrainedArtifact(
		method="HSRnet",
		family="HSRnet",
		framework="tensorflow",
		dataset_hint="Harvard",
		artifact_path="methods/_HSRnet/models(harvard)/model-150000.ckpt.index",
	),
	PretrainedArtifact(
		method="HSRnet",
		family="HSRnet",
		framework="tensorflow",
		dataset_hint="Harvard",
		artifact_path="methods/_HSRnet/models(harvard)/model-150000.ckpt.meta",
	),

	# DBIN (TensorFlow checkpoints)
	PretrainedArtifact(
		method="DBIN",
		family="EDBIN",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_DBIN/models_ibp_sn22/model-260000.ckpt.data-00000-of-00001",
		notes="TensorFlow v1-style checkpoint; see methods/_DBIN/train_cave_edbin.py",
	),
	PretrainedArtifact(
		method="DBIN",
		family="EDBIN",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_DBIN/models_ibp_sn22/model-260000.ckpt.index",
	),
	PretrainedArtifact(
		method="DBIN",
		family="BoostRes",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_DBIN/models_boost_res/model-250000.ckpt.index",
		notes="TensorFlow v1-style checkpoint; see methods/_DBIN/train_boost_res.py",
	),
	PretrainedArtifact(
		method="DBIN",
		family="BoostRes",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_DBIN/models_boost_res/model-250000.ckpt.meta",
	),
	PretrainedArtifact(
		method="DBIN",
		family="BoostRes",
		framework="tensorflow",
		dataset_hint="Harvard",
		artifact_path="methods/_DBIN/harvard code/models_boost_res_h/model-250000.ckpt.index",
		notes="TensorFlow v1-style checkpoint; see methods/_DBIN/harvard code/test_b_re_h.py",
	),
	PretrainedArtifact(
		method="DBIN",
		family="BoostRes",
		framework="tensorflow",
		dataset_hint="Harvard",
		artifact_path="methods/_DBIN/harvard code/models_boost_res_h/model-250000.ckpt.meta",
	),

	# MHFnet (TensorFlow checkpoint)
	PretrainedArtifact(
		method="MHFnet",
		family="CMHF-net",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_MHFnet/CMHF-net/temp/TrainedNet/model-epoch-30.data-00000-of-00001",
		notes="TensorFlow v1-style checkpoint; see methods/_MHFnet/CMHF-net/CAVEmain.py",
	),
	PretrainedArtifact(
		method="MHFnet",
		family="CMHF-net",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_MHFnet/CMHF-net/temp/TrainedNet/model-epoch-30.index",
	),

	# SpfNet (TensorFlow checkpoint)
	PretrainedArtifact(
		method="SpfNet",
		family="SpfNet",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_SpfNet/cave_data/SpfNet-net/model/-200.data-00000-of-00001",
		notes="TensorFlow v1-style checkpoint; see methods/_SpfNet/Spf.py",
	),
	PretrainedArtifact(
		method="SpfNet",
		family="SpfNet",
		framework="tensorflow",
		dataset_hint="CAVE",
		artifact_path="methods/_SpfNet/cave_data/SpfNet-net/model/-200.index",
	),

	# TSFN (pickle)
	PretrainedArtifact(
		method="TSFN",
		family="SSFSR",
		framework="pytorch",
		dataset_hint="CAVE",
		artifact_path="methods/_TSFN/models/ssfsr_9layers_epoch500.pkl",
		notes="Model pickle; see methods/_TSFN/test.py",
	),
]


def _iter_missing(artifacts: Iterable[PretrainedArtifact]) -> List[PretrainedArtifact]:
	return [artifact for artifact in artifacts if not artifact.exists()]


def main() -> int:
	parser = argparse.ArgumentParser(
		description="List pretrained weights present in this repository"
	)
	parser.add_argument("--list", action="store_true", help="Print a human-readable list")
	parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
	parser.add_argument(
		"--validate",
		action="store_true",
		help="Exit non-zero if any referenced file is missing",
	)
	args = parser.parse_args()

	if not (args.list or args.json or args.validate):
		args.list = True

	if args.json:
		print(
			json.dumps(
				[asdict(a) | {"exists": a.exists()} for a in ARTIFACTS],
				indent=2,
			)
		)

	if args.list:
		for a in ARTIFACTS:
			status = "OK" if a.exists() else "MISSING"
			print(
				f"[{status}] {a.method}/{a.family} ({a.framework}) "
				f"dataset≈{a.dataset_hint}: {a.artifact_path}"
			)
			if a.notes:
				print(f"        {a.notes}")

	if args.validate:
		missing = _iter_missing(ARTIFACTS)
		if missing:
			print(f"\nERROR: {len(missing)} referenced pretrained artifact(s) are missing.")
			for a in missing:
				print(f"- {a.artifact_path}")
			return 2

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
