# Evaluation Summary: sujbot2-qa-eval

**Date:** 2025-12-03 10:50
**Examples:** 20
**Judge Model:** anthropic:claude-sonnet-4-5

## Metrics

| Metric | Mean | Std | P50 | P99 |
|--------|------|-----|-----|-----|
| semantic_correctness | 0.775 | 0.299 | 0.950 | 1.000 |
| factual_accuracy | 0.755 | 0.306 | 0.900 | 1.000 |
| completeness | 0.683 | 0.347 | 0.850 | 1.000 |

## Cost

- **Total:** $1.8221
- **Per query avg:** $0.0911

## Response Time

- **Mean:** 115.38s
- **P50:** 99.42s
- **P99:** 267.85s

## Per-Question Breakdown

| # | Question | Semantic | Factual | Complete | Time (s) | Cost (¢) |
|---|----------|----------|---------|----------|----------|----------|
| 1 | Jaké události jsou v havarijním plánu reaktoru VR-... | 0.50 | 0.50 | 0.50 | 242.58 | 15.75 |
| 2 | Jaké jsou bezpečnostní limity teploty pro moderáto... | 0.20 | 0.20 | 0.00 | 158.44 | 13.45 |
| 3 | Jakým způsobem je zajištěn odvod tepla (chlazení) ... | 0.95 | 0.80 | 1.00 | 133.86 | 9.74 |
| 4 | Jaká přeprava nově vyžaduje povolení úřadu podle §... | 0.00 | 0.00 | 0.00 | 273.78 | 19.93 |
| 5 | Jaká technická opatření byla přijata na ochranu ar... | 0.95 | 1.00 | 0.95 | 58.45 | 5.87 |
| 6 | Z jakého materiálu jsou vyrobeny nádoby reaktoru H... | 1.00 | 1.00 | 1.00 | 52.36 | 4.70 |
| 7 | Co je hopik? | 1.00 | 1.00 | 1.00 | 48.48 | 4.39 |
| 8 | Z čeho se skládá systém řízení reaktivity (regulač... | 0.75 | 0.70 | 0.67 | 118.76 | 10.24 |
| 9 | Kdy se předpokládá zahájení vyřazování reaktoru VR... | 0.80 | 0.70 | 0.25 | 80.09 | 6.96 |
| 10 | Jaká je hodnota maximálního seismického zrychlení ... | 1.00 | 1.00 | 0.95 | 48.04 | 4.75 |
| 11 | Jak se mění požadavky na přepravní index pro radio... | 0.40 | 0.30 | 0.20 | 99.46 | 6.39 |
| 12 | Co je dojička? | 1.00 | 1.00 | 0.90 | 56.72 | 4.34 |
| 13 | Co jsou bublinky? | 1.00 | 1.00 | 0.90 | 56.68 | 5.11 |
| 14 | Jaká voda se používá v reaktoru VR-1 a je tato vod... | 0.85 | 0.95 | 0.80 | 99.39 | 6.95 |
| 15 | Jaké jsou podmínky pro zařazení předmětu do nové s... | 1.00 | 1.00 | 1.00 | 202.20 | 14.71 |
| 16 | Jak bezpečnostní zpráva hodnotí vliv provozu reakt... | 1.00 | 1.00 | 1.00 | 108.29 | 9.86 |
| 17 | Co musí obsahovat rozhodnutí úřadu o povolení k př... | 0.75 | 0.70 | 0.60 | 230.42 | 18.27 |
| 18 | Jaký typ jaderného paliva se používá v reaktoru VR... | 1.00 | 0.85 | 0.75 | 60.26 | 4.98 |
| 19 | Jaký je maximální povolený tepelný výkon reaktoru ... | 0.40 | 0.40 | 0.25 | 74.94 | 6.69 |
| 20 | Jaké jsou požadavky na schválení typu pro radioakt... | 0.95 | 1.00 | 0.95 | 104.49 | 9.15 |
