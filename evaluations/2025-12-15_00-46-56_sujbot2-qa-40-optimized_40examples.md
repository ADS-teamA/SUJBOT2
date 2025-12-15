# Evaluation Summary: sujbot2-qa-40-optimized

**Date:** 2025-12-15 00:46
**Examples:** 40
**Judge Model:** anthropic:claude-sonnet-4-5

## Metrics

| Metric | Mean | Std | P50 | P99 |
|--------|------|-----|-----|-----|
| semantic_correctness | 0.703 | 0.314 | 0.850 | 1.000 |
| factual_accuracy | 0.680 | 0.319 | 0.800 | 1.000 |
| completeness | 0.624 | 0.352 | 0.725 | 1.000 |

## Cost

- **Total:** $5.2768
- **Per query avg:** $0.1319

## Response Time

- **Mean:** 106.57s
- **P50:** 80.04s
- **P99:** 277.22s

## Per-Question Breakdown

| # | Question | Semantic | Factual | Complete | Time (s) | Cost (¢) |
|---|----------|----------|---------|----------|----------|----------|
| 1 | Jak moc vysoké je riziko výskytu povodně v oblasti... | 0.95 | 1.00 | 1.00 | 79.97 | 11.88 |
| 2 | Co to je jaderná položka? | 1.00 | 0.95 | 0.85 | 27.62 | 4.25 |
| 3 | Jaký je výrobce paliva používaného v reaktoru VR-1... | 1.00 | 1.00 | 0.95 | 38.51 | 4.87 |
| 4 | Jaký je maximální povolený tepelný výkon reaktoru ... | 0.50 | 0.40 | 0.25 | 82.45 | 8.35 |
| 5 | Jaké jsou podmínky pro zařazení předmětu do nové s... | 0.25 | 0.20 | 0.00 | 80.12 | 7.39 |
| 6 | Jaká přeprava nově vyžaduje povolení úřadu podle §... | 0.00 | 0.00 | 0.00 | 167.39 | 19.90 |
| 7 | Jaké jsou zodpovědné osoby reaktoru VR-1 a jaká je... | 0.60 | 1.00 | 0.80 | 52.00 | 5.31 |
| 8 | Kdy se předpokládá zahájení vyřazování reaktoru VR... | 0.85 | 0.50 | 0.25 | 52.05 | 6.78 |
| 9 | Co je hopík? | 1.00 | 0.95 | 1.00 | 36.96 | 4.48 |
| 10 | Co musí obsahovat rozhodnutí úřadu o povolení k př... | 0.75 | 0.55 | 0.65 | 211.76 | 18.79 |
| 11 | Jaká je hodnota maximálního seismického zrychlení ... | 1.00 | 1.00 | 0.85 | 35.79 | 4.77 |
| 12 | Jak zákon 263/2016 specifikuje záření pro lékařské... | 0.50 | 0.50 | 0.10 | 50.95 | 5.61 |
| 13 | Jaký je seznam všech firem, které jsou v roli exte... | 0.20 | 0.30 | 0.15 | 174.89 | 16.78 |
| 14 | Jaké jsou zákonné požadavky na dokumentaci nakládá... | 0.30 | 0.30 | 0.20 | 122.29 | 21.00 |
| 15 | Do jakého hlubinného úložiště je odvážen radioakti... | 0.85 | 0.75 | 0.65 | 213.13 | 20.78 |
| 16 | Jaké je členění radioaktivních zásilek a jak se li... | 0.45 | 0.50 | 0.30 | 252.00 | 34.38 |
| 17 | Je z pohledu zákona 263/2016 budova těžkých labora... | 1.00 | 0.75 | 0.95 | 237.99 | 59.02 |
| 18 | K čemu se využívá zkouška volným pádem a v čem spo... | 0.60 | 0.40 | 0.65 | 58.17 | 5.66 |
| 19 | Posuzuje dokumentace reaktoru všechny aspekty klim... | 0.20 | 0.10 | 0.50 | 292.85 | 47.37 |
| 20 | Jakým způsobem je zajištěn odvod tepla (chlazení) ... | 1.00 | 0.95 | 0.95 | 150.12 | 13.11 |
| 21 | Jaká voda se používá v reaktoru VR-1 a je tato vod... | 0.75 | 0.90 | 0.75 | 151.52 | 15.79 |
| 22 | Má ČVUT v Praze povinnost na základě zákona 18/199... | 0.75 | 0.85 | 0.50 | 200.46 | 43.91 |
| 23 | Co je dojička? | 1.00 | 1.00 | 1.00 | 31.67 | 4.42 |
| 24 | Z čeho se skládá systém řízení reaktivity (regulač... | 1.00 | 1.00 | 1.00 | 94.54 | 8.85 |
| 25 | Existují nějaká omezení pro materiály použité při ... | 1.00 | 1.00 | 1.00 | 43.67 | 4.70 |
| 26 | Jaký typ jaderného paliva se používá v reaktoru VR... | 0.90 | 0.75 | 0.70 | 32.25 | 5.02 |
| 27 | Jaké jsou požadavky na schválení typu pro radioakt... | 1.00 | 0.90 | 1.00 | 104.96 | 11.16 |
| 28 | Z jakého materiálu jsou vyrobeny nádoby reaktoru H... | 1.00 | 1.00 | 1.00 | 33.02 | 5.16 |
| 29 | Jak se mění požadavky na přepravní index pro radio... | 0.30 | 0.30 | 0.00 | 54.30 | 4.92 |
| 30 | Jaké události jsou v havarijním plánu reaktoru VR-... | 0.50 | 0.50 | 0.50 | 163.58 | 19.05 |
| 31 | Jaká je přesná délka protipovodňových bariér chrán... | 0.95 | 0.95 | 0.85 | 68.13 | 8.28 |
| 32 | Jaký je přesný vzorec pro výpočet bezpečné podkrit... | 0.50 | 0.30 | 0.40 | 44.19 | 5.00 |
| 33 | Jak zákon 263/2016 Sb. upravuje podmínky pro příje... | 0.00 | 0.10 | 0.00 | 115.63 | 7.77 |
| 34 | Co jsou bublinky? | 1.00 | 1.00 | 1.00 | 39.28 | 4.56 |
| 35 | Jaká povolení v rámci své odborné činnosti SÚJB mů... | 0.50 | 0.50 | 0.30 | 43.02 | 4.81 |
| 36 | Jak bezpečnostní zpráva hodnotí vliv provozu reakt... | 0.85 | 1.00 | 0.95 | 116.61 | 10.38 |
| 37 | Kteří pracovníci vykonávají na reaktoru VR-1 činno... | 1.00 | 0.90 | 0.95 | 62.47 | 7.45 |
| 38 | Jaké jsou bezpečnostní limity teploty pro moderáto... | 0.30 | 0.30 | 0.25 | 148.24 | 13.22 |
| 39 | Jaká technická opatření byla přijata na ochranu ar... | 0.95 | 1.00 | 0.95 | 45.41 | 4.85 |
| 40 | Jak atomový zákon 263/2016 Sb. upravuje trestnou s... | 0.85 | 0.85 | 0.80 | 252.78 | 17.90 |
