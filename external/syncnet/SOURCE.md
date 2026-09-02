# SyncNet — third-party model

Architecture and original checkpoint: Chung & Zisserman, ["Out of Time:
Automated Lip Sync in the Wild"](http://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/),
ACCV 2016 Workshop.

- Code mirrored from https://github.com/joonson/syncnet_python (MIT License).
- Checkpoint (`syncnet_v2.model`, ~55MB) is **not** committed to this repo.
  Download it from the original repository above, or from the mirror at
  https://huggingface.co/ByteDance/LatentSync-1.6/blob/main/auxiliary/syncnet_v2.model,
  and place it at `external/syncnet/syncnet_v2.model`.

Used here, unmodified and with no fine-tuning, purely as an off-the-shelf,
zero-training audio-visual synchrony feature source (Section 3.3 of the
paper).
