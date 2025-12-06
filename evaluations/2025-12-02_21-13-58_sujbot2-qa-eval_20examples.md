# Evaluation Summary: sujbot2-qa-eval

**Date:** 2025-12-02 21:13
**Examples:** 20
**Judge Model:** openai:gpt-4o-mini

## Metrics

| Metric | Mean | Std | P50 | P99 |
|--------|------|-----|-----|-----|
| semantic_correctness | 0.660 | 0.317 | 0.800 | 0.900 |
| factual_accuracy | 0.640 | 0.267 | 0.700 | 0.981 |
| completeness | 0.615 | 0.313 | 0.700 | 0.981 |

## Cost

- **Total:** $3.1563
- **Per query avg:** $0.1578

## Response Time

- **Mean:** 34.61s
- **P50:** 30.57s
- **P99:** 67.04s

## Per-Question Breakdown

| # | Question | Semantic | Factual | Complete | Time (s) | Cost (¢) |
|---|----------|----------|---------|----------|----------|----------|
| 1 | Jaký je maximální povolený tepelný výkon reaktoru ... | 0.70 | 0.70 | 0.40 | 29.42 | 1.50 |
| 2 | Jaký typ jaderného paliva se používá v reaktoru VR... | 0.90 | 0.90 | 0.70 | 23.70 | 2.82 |
| 3 | Jaká technická opatření byla přijata na ochranu ar... | 0.90 | 0.70 | 1.00 | 39.04 | 4.15 |
| 4 | Jakým způsobem je zajištěn odvod tepla (chlazení) ... | 0.90 | 0.70 | 0.90 | 32.15 | 5.44 |
| 5 | Z čeho se skládá systém řízení reaktivity (regulač... | 0.70 | 0.40 | 0.70 | 67.61 | 7.69 |
| 6 | Jaké události jsou v havarijním plánu reaktoru VR-... | 0.00 | 0.00 | 0.00 | 29.61 | 9.03 |
| 7 | Z jakého materiálu jsou vyrobeny nádoby reaktoru H... | 0.70 | 0.90 | 0.70 | 23.75 | 10.49 |
| 8 | Jaká je hodnota maximálního seismického zrychlení ... | 0.90 | 0.90 | 0.90 | 24.15 | 11.94 |
| 9 | Jaké jsou bezpečnostní limity teploty pro moderáto... | 0.00 | 0.00 | 0.00 | 26.41 | 13.31 |
| 10 | Kdy se předpokládá zahájení vyřazování reaktoru VR... | 0.90 | 0.70 | 0.70 | 31.82 | 14.60 |
| 11 | Jaká voda se používá v reaktoru VR-1 a je tato vod... | 0.90 | 0.90 | 0.90 | 40.95 | 16.47 |
| 12 | Co je hopik? | 0.80 | 0.70 | 0.90 | 23.62 | 17.79 |
| 13 | Co je dojička? | 0.90 | 0.70 | 0.90 | 25.88 | 19.11 |
| 14 | Jak bezpečnostní zpráva hodnotí vliv provozu reakt... | 0.40 | 0.70 | 0.40 | 30.49 | 20.46 |
| 15 | Co jsou bublinky? | 0.90 | 0.70 | 0.90 | 28.07 | 21.78 |
| 16 | Co musí obsahovat rozhodnutí úřadu o povolení k př... | 0.80 | 0.70 | 0.40 | 55.71 | 24.45 |
| 17 | Jaké jsou podmínky pro zařazení předmětu do nové s... | 0.40 | 0.40 | 0.40 | 32.78 | 25.79 |
| 18 | Jaká přeprava nově vyžaduje povolení úřadu podle §... | 0.00 | 0.40 | 0.10 | 64.60 | 28.30 |
| 19 | Jak se mění požadavky na přepravní index pro radio... | 0.60 | 0.70 | 0.50 | 30.64 | 29.59 |
| 20 | Jaké jsou požadavky na schválení typu pro radioakt... | 0.90 | 1.00 | 0.90 | 31.76 | 30.90 |
