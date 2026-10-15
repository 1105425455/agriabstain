# Development-selected operating point

Artifacts behind the development-selected threshold reported in the paper. The threshold is
chosen on development records only and then frozen and applied to the existing comprehensive
test predictions.

## Protocol

1. Run the SFT-native checkpoint on the canonical development set (2,408 records).
2. Extract the calibrated three-class gate probabilities on the same development set.
3. Take the highest empirical threshold at which ReGate development supported retention is
   at least the SFT-native development retention.
4. Freeze that threshold and apply it to the comprehensive test predictions.

Checkpoint, gate and threshold selection use no test or external-probe record.

## Data audit

- Development records: 2,408 (1,204 supported + 1,204 unsupported).
- All development sample IDs are unique.
- The canonical development split and `sft_dev.json` hold the same record multiset.
- Source-image overlap between development and test is 0.
- Gate probability extraction covers 2,408/2,408 development records with 0 errors.

## Results

| Quantity | Value |
|---|---|
| Development SFT-native supported retention | 0.8605 |
| Development-selected threshold | 0.9888232350349426 |
| ReGate development supported retention at that threshold | 0.8605 |
| Test SFT-native supported retention / FCA | 0.8511 / 0.0945 |
| Test ReGate supported retention / FCA at the frozen threshold | 0.8896 / 0.0454 |

With the threshold fixed on development data, test FCA falls 52.0% relative to SFT native
while supported retention rises from 0.8511 to 0.8896.

Per-threshold values are in `analysis/dev_threshold_sweep.csv`; the full record is in
`analysis/result.json`.
