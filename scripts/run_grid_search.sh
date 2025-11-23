#!/bin/bash
# Grid Search Runner - Spustí systematické testování všech kombinací parametrů

echo "================================="
echo "Grid Search Evaluation (k=100)"
echo "================================="
echo ""
echo "Konfigurace:"
echo "  - HyDE: [True, False]"
echo "  - num_expands: [0, 1, 2]"
echo "  - search_method: [hybrid, dense_only, bm25_only]"
echo "  - Celkem: 18 konfigurací"
echo ""
echo "Výsledky budou uloženy do: results/grid_search_k100/"
echo ""
read -p "Spustit grid search? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Zrušeno."
    exit 1
fi

echo ""
echo "Spouštím grid search v Docker kontejneru..."
docker exec sujbot_backend python /app/scripts/eval_grid_search.py

echo ""
echo "Grid search dokončen!"
echo "Kopíruji výsledky na host..."
docker cp sujbot_backend:/app/results/grid_search_k100 ./results/

echo ""
echo "✓ Výsledky zkopírovány do: ./results/grid_search_k100/"
echo "✓ Souhrnný report: ./results/grid_search_k100/grid_search_summary_k100.json"
