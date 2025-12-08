# Neo4j Bloom Setup pro SUJBOT2

Neo4j Bloom je interaktivní vizualizační nástroj pro exploraci grafových dat bez nutnosti psát Cypher dotazy. Je ideální pro prezentace stakeholderům a netechnické uživatele.

## Instalace

### 1. Stáhnout Neo4j Desktop

```bash
# Stáhnout z https://neo4j.com/download/
# Bloom je součástí Neo4j Desktop (zdarma)
```

### 2. Připojit k existující Neo4j instanci

1. Otevřít Neo4j Desktop
2. Kliknout na **"Add"** → **"Remote connection"**
3. Zadat údaje:
   - **Name**: SUJBOT2 Production
   - **Connect URL**: `bolt://localhost:7687` (nebo adresa vašeho serveru)
   - **Username**: `neo4j`
   - **Password**: (z `.env` souboru)

### 3. Otevřít Bloom

1. Vybrat připojenou databázi
2. Kliknout na **"Open"** → **"Neo4j Bloom"**

---

## Konfigurace Perspective

Perspective definuje, jak Bloom zobrazuje různé typy uzlů a vztahů.

### Vytvořit SUJBOT2 Perspective

1. V Bloom kliknout na **ikonu ozubeného kola** (vpravo nahoře)
2. Kliknout na **"Create Perspective"**
3. Pojmenovat: **"SUJBOT2 Knowledge Graph"**

### Nakonfigurovat kategorie uzlů

V editoru Perspective přidat následující kategorie:

#### Základní entity (Core)
| Label | Barva | Ikona |
|-------|-------|-------|
| `Entity` kde `type = 'organization'` | Modrá | 🏢 |
| `Entity` kde `type = 'person'` | Světle modrá | 👤 |
| `Entity` kde `type = 'location'` | Tyrkysová | 📍 |
| `Entity` kde `type = 'date'` | Zelená | 📅 |

#### Regulatorní hierarchie
| Label | Barva | Ikona |
|-------|-------|-------|
| `Entity` kde `type = 'regulation'` | Fialová | 📜 |
| `Entity` kde `type = 'decree'` | Tmavě fialová | 📋 |
| `Entity` kde `type = 'requirement'` | Světle fialová | ✅ |

#### Jaderně technické
| Label | Barva | Ikona |
|-------|-------|-------|
| `Entity` kde `type = 'reactor'` | Červená | ⚛️ |
| `Entity` kde `type = 'facility'` | Tmavě červená | 🏭 |
| `Entity` kde `type = 'system'` | Růžová | ⚙️ |

#### České právní typy
| Label | Barva | Ikona |
|-------|-------|-------|
| `Entity` kde `type = 'vyhlaska'` | Zelená | 📄 |
| `Entity` kde `type = 'narizeni'` | Tmavě zelená | 📑 |

---

## Search Phrases (Vyhledávací fráze)

Search phrases umožňují uživatelům vyhledávat v grafu přirozeným jazykem.

### Přidat Search Phrases

V Perspective editoru → **"Search phrases"** přidat:

#### 1. Hledání organizací
```
Find organization $name
```
**Cypher:**
```cypher
MATCH (e:Entity {type: 'organization'})
WHERE e.value CONTAINS $name
RETURN e
```

#### 2. Hledání regulací
```
Show regulations about $topic
```
**Cypher:**
```cypher
MATCH (e:Entity {type: 'regulation'})-[:covers_topic]->(t:Entity {type: 'topic'})
WHERE t.value CONTAINS $topic
RETURN e, t
```

#### 3. Hledání požadavků pro facility
```
Requirements for $facility
```
**Cypher:**
```cypher
MATCH (f:Entity {type: 'facility'})-[:regulated_by]->(r:Entity)-[:specifies_requirement]->(req:Entity {type: 'requirement'})
WHERE f.value CONTAINS $facility
RETURN f, r, req
```

#### 4. Všechny entity z dokumentu
```
Entities from document $doc_id
```
**Cypher:**
```cypher
MATCH (e:Entity)
WHERE e.document_id CONTAINS $doc_id
RETURN e
LIMIT 100
```

#### 5. Compliance gaps (nesoulady)
```
Show compliance gaps
```
**Cypher:**
```cypher
MATCH (gap:Entity {type: 'compliance_gap'})-[r]->(req:Entity {type: 'requirement'})
RETURN gap, r, req
```

#### 6. Vztahy mezi dvěma entitami
```
Connection between $entity1 and $entity2
```
**Cypher:**
```cypher
MATCH path = shortestPath(
  (a:Entity)-[*..5]-(b:Entity)
)
WHERE a.value CONTAINS $entity1 AND b.value CONTAINS $entity2
RETURN path
```

---

## Styling pravidla

### Velikost uzlů podle confidence
1. V Perspective editoru → **"Rules"**
2. Přidat pravidlo:
   - **Condition**: `confidence > 0.8`
   - **Style**: Size = Large
3. Přidat pravidlo:
   - **Condition**: `confidence < 0.5`
   - **Style**: Size = Small, Color = Gray

### Zvýraznění důležitých vztahů
1. Přidat pravidlo pro hrany:
   - **Relationship type**: `complies_with`
   - **Style**: Color = Green, Width = Thick
2. Přidat pravidlo:
   - **Relationship type**: `contradicts`
   - **Style**: Color = Red, Width = Thick

---

## Běžné úkoly v Bloom

### Explorovat okolí entity
1. Najít entitu pomocí search phrase
2. Double-click na uzel → **"Expand"**
3. Vybrat typy vztahů k zobrazení

### Filtrovat zobrazené entity
1. Kliknout na **ikonu filtru** (vlevo)
2. Vybrat typy entit k zobrazení/skrytí
3. Nastavit rozsah confidence

### Export vizualizace
1. Kliknout na **ikonu stažení** (vpravo nahoře)
2. Vybrat formát: PNG nebo SVG
3. Nastavit rozlišení pro prezentace

### Sdílet pohled
1. Uložit aktuální scene jako **"Saved scene"**
2. Sdílet scene s kolegy (vyžaduje stejnou Perspective)

---

## Tipy pro prezentace

### Před prezentací
1. Připravit několik **Saved scenes** s klíčovými pohledy
2. Otestovat search phrases
3. Nastavit vhodné barvy a velikosti

### Během prezentace
1. Používat **full-screen mode** (F11)
2. Používat **"Clear scene"** mezi tématy
3. Double-click pro expand, single-click pro select

### Klávesové zkratky
| Zkratka | Akce |
|---------|------|
| `Escape` | Zrušit výběr |
| `Delete` | Skrýt vybrané uzly |
| `Ctrl+A` | Vybrat vše |
| `Ctrl+Z` | Zpět |
| `+/-` | Přiblížit/Oddálit |

---

## Řešení problémů

### Bloom se nepřipojí k databázi
1. Ověřit, že Neo4j běží: `docker compose ps`
2. Ověřit credentials v `.env`
3. Zkontrolovat firewall (port 7687)

### Prázdný graf
1. Ověřit, že graf obsahuje data:
   ```cypher
   MATCH (n) RETURN count(n)
   ```
2. Zkontrolovat label `Entity` (Bloom hledá specifické labely)

### Pomalá odezva
1. Omezit počet zobrazených uzlů (použít `LIMIT` v search phrases)
2. Zakázat automatický expand
3. Použít filtrování podle typu entity

---

## Další zdroje

- [Neo4j Bloom Documentation](https://neo4j.com/docs/bloom-user-guide/)
- [Bloom Perspectives Guide](https://neo4j.com/docs/bloom-user-guide/current/bloom-perspectives/)
- [Search Phrases Tutorial](https://neo4j.com/docs/bloom-user-guide/current/bloom-search-phrases/)
