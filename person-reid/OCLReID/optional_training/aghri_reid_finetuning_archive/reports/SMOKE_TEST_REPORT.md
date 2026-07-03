# SMOKE TEST REPORT

CUDA available: False

This smoke test used a synthetic PK-like CPU batch to verify model mechanics.
It did not train the Stage 1 model.

## Results

- Compatible checkpoint keys loaded: 134
- Ignored unexpected keys: 0
- Shape mismatches: 2
- Finite loss: True
- Total synthetic loss: 3.522760
- Trainable tensors with gradients: 40
- Frozen gradient violations: 0

## Loss Dictionary

```json
{
  "triplet_loss": 2.108691453933716,
  "ce_loss": 1.4140690565109253,
  "accuracy": "{'top-1': tensor([8.3333])}"
}
```

## Shape Mismatches

```json
[
  [
    "head.classifier.weight",
    [
      380,
      128
    ],
    [
      3,
      128
    ]
  ],
  [
    "head.classifier.bias",
    [
      380
    ],
    [
      3
    ]
  ]
]
```

## Decision

Smoke test PASS.

Full Stage 1 training was not launched because CUDA is not available in the
current `oclreid` environment.
